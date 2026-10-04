"""P1 · 遮挡敏感度（Omission）：99.73% 的模型到底盯图里的哪一块？

🔴 为什么必须做：
    数字（99.73% macro-F1 / 1,912 张只错 6 张）高得反常，
    而「高精度」有三种完全不同的成因，后两种对落地是致命的：
      (a) 真的盯住了缺陷             → 好
      (b) 盯住**打印件几何/整体形态**（翘曲/离台本来就是整体形变）
          → 只能识别「这批打印件」，换机型/换材料即失效
      (c) 盯住**背景/拍摄条件**       → 已排除（干预实验仅 −0.33pp）
    区分 (a) 与 (b) 只能靠「遮挡后掉多少分」：
    缺陷是**局部**现象，遮住它应该明显掉分；
    整体形态是**分布式**的，遮任何一块都只轻微掉分。

🔴 三条实现纪律（都是踩出来的）：
  1. **不用 Grad-CAM**。L2 是 LogisticRegression 线性探针，没有梯度可借；
     硬凑 Grad-CAM 只能对 L3 微调权重做，而 L3 已证明无增益（−0.54pp），
     拿一个「证明无增益的模型」去解释「好成绩」是错的。
     遮挡敏感度对**任意黑盒**都成立，且不需要猜模块名
     （EfficientNet 的子模块是 `blocks` 而不是 ResNet 的 `layer4`）。
  2. **指标要连续，不能用 0/1 翻转**。5 分类里「错一类」只有 0 和 1 两档，
     16 个格子全是 0 只能说明「没翻转」，不能说明「没信息」。
     → 记录**真值类概率的相对降幅** `rel_drop = (p0 - p) / p0`，连续、可平均。
  3. **每类多取几张再平均**。单张图的遮挡热区有偶然性，
     每类取 5 张（第 0 折内）做平均，才敢下结论。

输出：
  exp/p1_d3_occlusion.json热区统计
  exp/p1_occlusion_grid.png    代表图 + 敏感块热区
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exp" / "p1_d3_occlusion.json"
PNG = ROOT / "exp" / "p1_occlusion_grid.png"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
IMG = 224
GRID = 4          # 切成 4x4 = 16 块，每块 56×56
PER_CLS = 5# 每类取几张代表图求平均


def load_fold0():
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    f = np.array([x["fold_grouped"] for x in items])
    return items, np.flatnonzero(f == 0)


def fit_l2(F, y, items):
    """在除第 0 折以外的数据上拟合 L2 线性探针（与 train.py 的 L2 同配方）"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    f = np.array([x["fold_grouped"] for x in items])
    tr = np.flatnonzero(f != 0)
    sc = StandardScaler().fit(F[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(F[tr]), y[tr])
    return sc, clf


def main():
    items, te = load_fold0()
    F = np.load(ROOT / "exp" / "_cache" / "l2_feats.npy")
    y = np.array([CLASSES.index(x["label"]) for x in items])
    sc, clf = fit_l2(F, y, items)

    import torch
    from torchvision import transforms
    from src.vision.weights import build_model
    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406),
                                                  (0.229, 0.224, 0.225))])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("efficientnet_b0", num_classes=0, img_size=IMG).eval().to(dev)

    @torch.no_grad()
    def prob_of(imgs, true_idx):
        """一次批量前向，返回每张图在真值类上的概率。imgs 必须是 PIL 对象。"""
        xs = [tf(im) for im in imgs]
        with torch.amp.autocast(dev, enabled=(dev == "cuda")):
            fr = model.forward_features(torch.stack(xs).to(dev))
        fv = fr.float().mean(dim=(2, 3)).cpu().numpy()
        prob = clf.predict_proba(sc.transform(fv))
        return prob[np.arange(len(imgs)), true_idx]

    def load_rgb(p):
        with Image.open(p) as im:
            return im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)

    # 每类取 PER_CLS 张代表图（第 0 折内，全图预测正确的）
    picks = {}
    for ci, c in enumerate(CLASSES):
        idx = [i for i in te if y[i] == ci]
        picks[c] = idx[:PER_CLS]
        assert picks[c], f"{c} 在第 0 折里没有样本"

    print("=" * 70)
    print(f"遮挡敏感度：{GRID}×{GRID} 逐块涂黑，每类 {PER_CLS} 张求平均")
    print("指标 = 真值类概率的相对降幅 (p0-p)/p0，连续值，不是 0/1 翻转")
    heat, flips = {}, {}
    for cls in CLASSES:
        ci = CLASSES.index(cls)
        paths = [items[i]["path"] for i in picks[cls]]
        p0 = prob_of([load_rgb(p) for p in paths], ci)
        drops = np.zeros((GRID, GRID))
        nflip = np.zeros((GRID, GRID))
        for r in range(GRID):
            for c in range(GRID):
                box = (c * IMG // GRID, r * IMG // GRID,
                       (c + 1) * IMG // GRID, (r + 1) * IMG // GRID)
                masked = []
                for p in paths:
                    m = load_rgb(p)
                    m.paste((0, 0, 0), box)
                    masked.append(m)
                p = prob_of(masked, ci)
                rel = (p0 - p) / np.maximum(p0, 1e-6)
                drops[r, c] = np.clip(rel, 0, None).mean()
                nflip[r, c] = (p < p0).mean()
        heat[cls] = drops
        flips[cls] = nflip
        print(f"  {cls:<15} 真值类原始概率 {p0.mean():.3f}→"
              f"最大降幅块 {drops.max():.3f} | 全图降幅均值 {drops.mean():.4f}")

    # 量化：敏感度是否集中在中心（缺陷通常在打印件中部）
    stats = {}
    for cls in CLASSES:
        h = heat[cls]
        inner = h[1:GRID - 1, 1:GRID - 1].sum()
        tot = h.sum()
        # 集中度：热区基尼式度量——最高块占总量比例
        stats[cls] = {
            "n_flipped_blocks_mean": round(float(flips[cls].sum()), 2),
            "rel_drop_max": round(float(h.max()), 4),
            "rel_drop_mean": round(float(h.mean()), 4),
            "center_ratio": round(float(inner / tot), 3) if tot > 1e-9 else None,
            "top_share": round(float(h.max() / tot), 3) if tot > 1e-9 else None,
            "rel_drop_map": np.round(h, 4).tolist(),
            "flip_map": np.round(flips[cls], 3).tolist(),
        }

    print("\n" + "-" * 70)
    print("相对降幅热区（×100，数值越大= 该块承载的类别信息越多）：")
    hdr = "".join(f"{c:>7}" for c in range(GRID))
    for cls in CLASSES:
        print(f"\n  {cls}:")
        print(f"      {hdr}")
        for r in range(GRID):
            print("      " + "".join(f"{v*100:>7.1f}" for v in heat[cls][r]))

    print("\n" + "=" * 70)
    print("关键判定：")
    zero_cls = [c for c in CLASSES if heat[c].max() < 0.02]
    if zero_cls:
        print(f"  ⚠️ {zero_cls} 遮任何一块都不掉分（max<2%）")
    strong = {c: stats[c]["rel_drop_max"] for c in CLASSES}
    ranked = sorted(strong.items(), key=lambda kv: -kv[1])
    print("  单块最大降幅排名：")
    for c, v in ranked:
        print(f"    {c:<15} {v*100:>6.1f}%")
    spread = {c: float((heat[c] > 0.2 * heat[c].max()).sum()) for c in CLASSES}
    print("\n  「有效块」数（降幅 > 20% 自身峰值 的块数，共 16）：")
    for c, v in spread.items():
        print(f"    {c:<15} {int(v):>2d} 块")
    conc = np.mean([spread[c] for c in CLASSES])
    print(f"\n  平均有效块数 = {conc:.1f}/16"
          f"（1~2 块=高度集中于局部；>10 块=信息分布式）")

    # 画图：代表图 + 热区
    canvas = Image.new("RGB", (len(CLASSES) * 220, 245), "white")
    dr = ImageDraw.Draw(canvas)
    for j, cls in enumerate(CLASSES):
        gi = picks[cls][0]
        base = load_rgb(items[gi]["path"])
        b = np.asarray(base, dtype=np.float32).copy()
        hm = heat[cls]
        cell = 220 // GRID
        vmax = max(hm.max(), 1e-9)
        for r in range(GRID):
            for c in range(GRID):
                if hm[r, c] > 0.05 * vmax:
                    y0, x0 = r * cell, c * cell
                    w = min(1.0, hm[r, c] / vmax)
                    b[y0:y0 + cell, x0:x0 + cell] = (
                        b[y0:y0 + cell, x0:x0 + cell] * (1 - 0.7 * w)
                        + np.array([255, 0, 0], dtype=np.float32) * 0.7 * w)
        canvas.paste(Image.fromarray(b.astype("uint8")), (j * 220, 0))
        dr.text((j * 220 + 3, 224),
                f"{cls} {hm.max()*100:.0f}%max", fill="red")
    canvas.save(PNG)

    OUT.write_text(json.dumps(
        {"grid": GRID, "img": IMG, "per_class": PER_CLS,
         "explained_fold": 0, "classifier": "L2 (StandardScaler+LogReg)",
         "metric": "rel_drop = (p0 - p_masked) / p0，真值类概率相对降幅",
         "stats": stats,
         "mean_effective_blocks": round(float(conc), 2),
         "note": ("遮挡敏感度而非 Grad-CAM：L2 是线性探针无梯度可借，"
                  "且 L3 已证无增益，拿无增益模型解释好成绩是错的。"
                  "遮挡对任意黑盒成立，模块名也不必猜。")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)} / {PNG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())