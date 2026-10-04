"""P1 · DINOv2 特征相似度分布标定（阈值自标定，**不照抄文献默认值**）

为什么必须做：
    文献默认 `cos ≥ 0.92` 在本数据集上**严重过松**——实测把 1,912 张压成 39 个 group
    （折叠 1,873 张），`StratifiedGroupKFold` 直接失效（一个 group 跨折 → 无法分层）。
    反过来 Warping 同类 8 张实测 min=0.931/mean=0.957，说明同类图像天然高相似。
    **阈值必须由本数据集的真实分布 + 人工抽查决定。**

产出：
    exp/p1_dedup_calibration.json  —— 各候选阈值下的 group 数/最大组/跨类污染率
    /tmp/dino_feats.npy            —— 特征缓存（提一次，后续调阈值秒级）

用法：
    python -m src.vision.calibrate_dedup --data-root D:/datasets/fdm-defect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
# ⚠️ 2026-10-04 实测坑：原生 Windows Python 认不了 Git Bash 的 `/tmp`，
#   `np.save("/tmp/x.npy")` 会解析成 `\tmp\x.npy`（不存在）→ FileNotFoundError，
#   而前面 1,912 张特征已提完（~5 min）全部白费。
#   **Python 代码里一律用项目内相对路径**，别用 /tmp。
CACHE_DIR = Path("exp/_cache")
FEAT_CACHE = CACHE_DIR / "dino_feats.npy"
PATH_CACHE = CACHE_DIR / "dino_paths.npy"

# 候选阈值：覆盖从严到松的范围。0.999 以上必为同一实物或近邻重拍。
CANDIDATES = [0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 0.99995]


def extract(root: Path, device: str = "cuda", batch_log: int = 200):
    """提 DINOv2 特征并缓存。逐张推理 1,912 张约 3~5 min（518px）。"""
    from PIL import Image
    from src.vision.dedup import FeatureExtractor

    # ⚠️ 预检：缓存目录**必须在提特征之前**就写通。
    #   上一次实跑 1,912 张特征提完（~5 min）才在 np.save 抛 FileNotFoundError，
    #   全部白算——因为「路径可写」这件事没法事后补救，只能事前验证。
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    probe = CACHE_DIR / "_write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print(f"[calib] 缓存目录可写: {CACHE_DIR.resolve()}")

    fx = FeatureExtractor(device=device)
    paths = []
    for c in CLASSES:
        paths += sorted((root / c).glob("*.jpg"))
    print(f"[calib] 待提特征 {len(paths)} 张，输入 {fx.input_size}")

    # 逐张推理 1,912 张约 5 min；`embed_batch` 分批（fp16）约 40 s。
    # 归一化与尺寸逻辑全在 FeatureExtractor 里（`to_batch`），**不在这里重复实现**
    # ——上一版就是在这里重写了一遍并引用了不存在的 `fx.torchvision`，白跑一次。
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB").resize(fx.input_size))
    print(f"[calib] 分批推理中（batch=16）...")
    F = fx.embed_batch(imgs, batch=16)
    del imgs
    print(f"[calib] 特征 {F.shape}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEAT_CACHE, F)
    np.save(PATH_CACHE, np.array([str(p) for p in paths], dtype=object))
    print(f"[calib] 特征已缓存 {F.shape} → {FEAT_CACHE}")
    return F, [str(p) for p in paths]


class _U:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def labels_of(paths):
    y = np.empty(len(paths), dtype=np.int64)
    for i, p in enumerate(paths):
        y[i] = CLASSES.index(Path(p).parent.name)
    return y


def group_stats(F, y, thr):
    n = len(F)
    S = F @ F.T
    uf = _U(n)
    ii, jj = np.triu_indices(n, 1)
    hit = S[ii, jj] >= thr
    for a, b in zip(ii[hit].tolist(), jj[hit].tolist()):
        uf.union(a, b)
    g = np.array([uf.find(i) for i in range(n)])
    uniq, cnt = np.unique(g, return_counts=True)
    # 跨类污染：一个 group 里出现多个类别 = 误把不同类当同一实物
    impure = 0
    for u in uniq:
        if len(np.unique(y[g == u])) > 1:
            impure += 1
    return {
        "threshold": round(float(thr), 5),
        "n_pairs": int(hit.sum()),
        "n_groups": int(len(uniq)),
        "folded": int(n - len(uniq)),
        "max_group": int(cnt.max()),
        "max_group_ratio": round(float(cnt.max() / n), 4),
        "n_multi_groups": int((cnt > 1).sum()),
        "n_impure_groups": int(impure),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/datasets/fdm-defect")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="复用已有特征缓存")
    a = ap.parse_args()

    if a.reuse and FEAT_CACHE.exists():
        F = np.load(FEAT_CACHE)
        paths = list(np.load(PATH_CACHE, allow_pickle=True))
        print(f"[calib] 复用缓存特征 {F.shape}")
    else:
        from src.vision.guards import probe_dataset
        root = probe_dataset(Path(a.data_root), strict=False).root
        F, paths = extract(root, a.device)

    y = labels_of(paths)
    S = F @ F.T
    iu = np.triu_indices(len(F), 1)
    v = S[iu]
    print(f"\n[calib] 两两余弦分布（共 {len(v):,} 对）")
    dist = {}
    for q in [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]:
        dist[f"P{q}"] = round(float(np.percentile(v, q)), 5)
        print(f"  P{q:<6} = {dist[f'P{q}']}")

    # 同类 vs 跨类 的可分性——判断阈值该落在哪
    same = y[iu[0]] == y[iu[1]]
    print(f"\n[calib] 同类对 {same.sum():,} 均值 {v[same].mean():.4f} | "
          f"跨类对 {(~same).sum():,} 均值 {v[~same].mean():.4f}")

    rows = [group_stats(F, y, t) for t in CANDIDATES]
    print(f"\n{'阈值':>10} {'组数':>6} {'折叠':>6} {'最大组':>7} {'占比':>7} "
          f"{'多图组':>7} {'跨类组':>7}")
    for r in rows:
        print(f"{r['threshold']:>10} {r['n_groups']:>6} {r['folded']:>6} "
              f"{r['max_group']:>7} {r['max_group_ratio']:>7.1%} "
              f"{r['n_multi_groups']:>7} {r['n_impure_groups']:>7}")

    out = {
        "n_images": int(len(F)),
        "feature_dim": int(F.shape[1]),
        "percentiles": dist,
        "same_class_mean": round(float(v[same].mean()), 5),
        "cross_class_mean": round(float(v[~same].mean()), 5),
        "candidates": rows,
        "note": "阈值需人工抽查样本后定稿；不要直接照抄文献 0.92。",
    }
    Path("exp").mkdir(exist_ok=True)
    Path("exp/p1_dedup_calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[产出] exp/p1_dedup_calibration.json")


if __name__ == "__main__":
    main()
