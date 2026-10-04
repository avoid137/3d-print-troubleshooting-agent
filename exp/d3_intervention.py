"""P1 · D3 决定性对照：98.44% 里有多少是「缺陷形态」，多少是「背景/批次捷径」？

🔴 审计结论（exp/audit_d3_credibility.py 实测）：
    仅凭低级像素统计（RGB 均值/直方图/梯度，**无空间结构**）
    就能拿到 macro-F1 = **71.5%**。
    各类背景色两两分明：
      Warping / Off_platform      R≈0.356G≈0.61  → 绿打印床 + 金色件
      Cracking / Stringing / Layer_shifting  R≈0.44 G≈0.46 → 灰白背景
    ⇒ L1 的98.44% 很可能主要来自「背景色 + 打印件颜色」，
      **不等于**缺陷形态识别能力。

本脚本设计 3 个干预级别，逐一剥离非缺陷线索：
    V0 原始        ：Resize(224) 全图——D3 现状
    V1 中心裁剪    ：中心 60% 区域放大到 224 —— 削弱背景占比
    V2 去背景归一化：把**边缘边框**（估计为纯背景）的颜色统计量从每通道
                     减掉，再截断到 [0,1] —— 强制消除一阶颜色偏移
    V3 灰度+裁剪   ：去色 + 中心裁剪 —— 彻底移除颜色捷径，只留形态

每个级别都跑 L1(kNN) + L2(LogReg)，group-aware 口径，5 折 seed=42。
若 V0 ≫ V3，说明高分数**主要**来自颜色/背景，**不能**对外宣称
「缺陷分类准确率 98%」；能对外宣称的只有 V3 这个「形态下界」。

产物：exp/p1_d3_intervention.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "exp" / "_cache"
OUT = ROOT / "exp" / "p1_d3_intervention.json"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
IMG = 224


# ------------------------------------------------------------ 干预算子
def _load(p):
    with Image.open(p) as im:
        im.draft("RGB", (IMG * 2, IMG * 2))
        return im.convert("RGB").resize((IMG * 2, IMG * 2), Image.BILINEAR).copy()


def v0_raw(im):
    return im.resize((IMG, IMG), Image.BILINEAR)


def v1_center(im, frac=0.60):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch)).resize((IMG, IMG), Image.BILINEAR)


def v2_debg(im, border=0.10):
    """按边缘边框的**中位数**做一阶颜色平移，消除背景色偏移。"""
    a = np.asarray(im, dtype=np.float32) / 255.0
    h, w, _ = a.shape
    bh, bw = int(h * border), int(w * border)
    edge = np.concatenate([
        a[:bh].reshape(-1, 3), a[-bh:].reshape(-1, 3),
        a[:, :bw].reshape(-1, 3), a[:, -bw:].reshape(-1, 3)])
    med = np.median(edge, axis=0)                # 每通道背景基准
    out = np.clip(a - med + 0.5, 0, 1)           # 平移到中性灰
    return Image.fromarray((out * 255).astype(np.uint8)).resize(
        (IMG, IMG), Image.BILINEAR)


def v3_gray(v2img):
    return v2img.convert("L").convert("RGB")


VARIANTS = {
    "V0_raw":        {"fn": v0_raw, "desc": "原始全图（当前 D3 口径）"},
    "V1_center60":   {"fn": v1_center, "desc": "中心 60% 裁剪（削弱背景）"},
    "V2_debg":       {"fn": v2_debg, "desc": "边缘边框中位数去背景平移"},
    "V3_gray_debg":  {"fn": lambda im: v3_gray(v2_debg(im)),
                      "desc": "去色 + 去背景（只留形态）"},
}


def build(items, key, mean, std, cache_tag):
    cp = CACHE / f"x_{cache_tag}.npy"
    if cp.exists():
        return np.load(cp)
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    fn = VARIANTS[key]["fn"]
    arr = np.zeros((len(items), 3, IMG, IMG), dtype=np.float32)
    for i, it in enumerate(items):
        arr[i] = tf(fn(_load(it["path"])))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cp, arr)
    print(f"    缓存 {cp.name} {arr.shape}")
    return arr


def main():
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    y = np.array([CLASSES.index(x["label"]) for x in items])
    f = np.array([x["fold_grouped"] for x in items])   # 只跑主口径

    from src.vision.guards import audit_backbone
    from src.vision.weights import build_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import torch

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = build_model("efficientnet_b0", num_classes=0, img_size=IMG)
    model = model.eval().cuda()

    report = {"n": len(items), "scheme": "grouped", "variants": {}}
    for key, meta in VARIANTS.items():
        t0 = time.time()
        print(f"\n[{key}] {meta['desc']}")
        X = build(items, key, mean, std, key)
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                xb = torch.from_numpy(X[i:i + 64]).cuda()
                with torch.amp.autocast("cuda"):
                    ftr = model.forward_features(xb)
                feats.append(ftr.float().mean(dim=(2, 3)).cpu()
                             if ftr.dim() == 4 else ftr.float().cpu())
        F = torch.cat(feats).numpy()

        row = {"desc": meta["desc"]}
        for lvl in ("L1", "L2"):
            f1s, preds_all, ys_all = [], [], []
            for k in range(5):
                te = np.flatnonzero(f == k)
                tr = np.flatnonzero(f != k)
                if lvl == "L1":
                    Xtr, Xte = F[tr], F[te]
                    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
                else:
                    s = StandardScaler().fit(F[tr])
                    Xtr, Xte = s.transform(F[tr]), s.transform(F[te])
                    clf = LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")
                clf.fit(Xtr, y[tr])
                p = clf.predict(Xte)
                f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
                if lvl == "L2":
                    preds_all.append(p)
                    ys_all.append(y[te])
            row[lvl] = {
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "macro_f1_std": round(float(np.std(f1s)), 4),
                "folds": [round(float(v), 4) for v in f1s],
            }
            if lvl == "L2":
                rep = classification_report(np.concatenate(ys_all),
                                            np.concatenate(preds_all),
                                            target_names=CLASSES,
                                            zero_division=0, output_dict=True)
                row["per_class_f1"] = {c: round(float(rep[c]["f1-score"]), 4)
                                       for c in CLASSES}
        row["seconds"] = round(time.time() - t0, 1)
        report["variants"][key] = row
        print(f"    L1 kNNmacro-F1 = {row['L1']['macro_f1_mean']:.4f} ± "
              f"{row['L1']['macro_f1_std']:.4f}")
        print(f"    L2 LogReg     = {row['L2']['macro_f1_mean']:.4f} ± "
              f"{row['L2']['macro_f1_std']:.4f}   ({row['seconds']}s)")
        print(f"    L2 各类 F1: {row['per_class_f1']}")

    # ---------------- 结论 ----------------
    v0 = report["variants"]["V0_raw"]["L2"]["macro_f1_mean"]
    v3 = report["variants"]["V3_gray_debg"]["L2"]["macro_f1_mean"]
    gap = round(100 * (v0 - v3), 2)
    report["conclusion"] = {
        "V0_L2": v0, "V3_L2": v3, "gap_pp": gap,
        "verdict": ("形态下界" if gap >= 15 else
                    "部分来自背景" if gap >= 5 else "高分数主要来自形态"),
        "note": ("对外只能宣称 V3 这个「形态下界」。V0 含背景/颜色捷径，"
                 "不能表述为「缺陷识别准确率」。"),
    }
    print("\n" + "=" * 70)
    print(f"结论：V0(原始) L2 = {v0:.4f}→ V3(去色去背景) L2 = {v3:.4f}")
    print(f"      背景/颜色贡献 = **{gap:+.2f} pp** → {report['conclusion']['verdict']}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
