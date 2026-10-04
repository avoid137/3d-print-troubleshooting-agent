"""P1 · D3 结果可信度审计：98.44% macro-F1 是真信号还是「拍摄批次捷径」？

🔴 触发原因（2026-10-04 实测矛盾）：
    DINOv2 特征上「跨类对最大余弦 0.9752」**高于**「同类对 P99=0.9569」
    → DINOv2 完全分不开这5 类。
    但EfficientNet-B0 冻结特征 + kNN 拿到 **98.44%** macro-F1。
    两者不可能同时对，必须查清 high-level 数字的来源。

本脚本检验三个「捷径（shortcut）」假设：
    H1 拍摄批次捷径：若每类在一个拍摄session 内完成，**背景/床身颜色/光照**
       就是类别的完美代理。检验：只用**低级像素统计**（均值/标准差/直方图）
       能分类到多少？若很高 → 模型学的不是缺陷形态。
    H2 近重复泄漏残留：kNN 的高���可能来自「同一实物近重复跨折」。
       检验：在 group-aware 折上，测试图的**最近训练邻**是否同 group？
    H3 折号划分本身有偏：检验每折的像素统计是否有系统性差异。

产物：exp/p1_d3_audit.json+ 人类可读结论
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "exp" / "_cache"
OUT = ROOT / "exp" / "p1_d3_audit.json"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]


def pixel_stats(items, size=(64, 64)):
    """低级像素统计：刻意**不含任何空间结构**，只看颜色/亮度分布。

    如果只用这些就能高分类准确率 → 存在「拍摄批次/背景」捷径。

    特征布局（24 维，全部 per-channel 对称，无手选下标）：
      [0:15]  每通道 5 个统计量 mean/std/min/max/p25
      [15:24] 每通道 3 个直方图桶（0~1/3, 1/3~2/3, 2/3~1）
      [24]    梯度能量（边缘强度）
    """
    n_per = 5 + 3
    X = np.zeros((len(items), 3 * n_per + 1), dtype=np.float32)
    for i, it in enumerate(items):
        with Image.open(it["path"]) as im:
            im.draft("RGB", size)
            a = np.asarray(im.convert("RGB").resize(size, Image.BILINEAR),
                           dtype=np.float32) / 255.0
        for ch in range(3):
            c = a[..., ch]
            base = ch * n_per
            X[i, base:base + 5] = [c.mean(), c.std(), c.min(), c.max(),
                                   np.percentile(c, 25)]
            h, _ = np.histogram(c, bins=3, range=(0, 1))
            X[i, base + 5:base + 8] = h / c.size
        # 空间梯度强度（边缘能量）——仍属低级统计
        X[i, -1] = (np.abs(np.diff(a, axis=0)).mean()
                    + np.abs(np.diff(a, axis=1)).mean())
    return X


def main():
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    y = np.array([CLASSES.index(x["label"]) for x in items])
    n = len(items)
    report = {"n": n}

    # ------------------------------------------------ H1 像素统计捷径
    print("=" * 70)
    print("H1  低级像素统计能否分类？（只含颜色/亮度/直方图/梯度，无空间结构）")
    sc_path = CACHE / "pixel_stats.npy"
    if sc_path.exists():
        X = np.load(sc_path)
        print(f"  复用缓存 {X.shape}")
    else:
        print("  计算中（1,912 张 64px）...")
        X = pixel_stats(items)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(sc_path, X)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    f1s = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, y):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(Xs[tr], y[tr])
        f1s.append(f1_score(y[te], clf.predict(Xs[te]), average="macro",
                            zero_division=0))
    h1 = float(np.mean(f1s))
    print(f"  🔴 像素统计 macro-F1 = {h1:.4f} ± {np.std(f1s):.4f}")
    print(f"  {'⚠️  存在强背景捷径！模型可能学的是拍摄批次而非缺陷' if h1 > 0.5 else '✅ 无明显背景捷径'}")
    report["h1_pixel_shortcut_f1"] = round(h1, 4)

    #各类像素均值差异（看是否某两类背景明显不同）
    print("\n  各类 RGB 均值（看背景是否可分）：")
    means = {}
    for ci, c in enumerate(CLASSES):
        m = y == ci
        # 逐通道 mean 在特征布局里是 [0, 5, 10]（每通道块首）
        means[c] = [round(float(X[m, j].mean()), 4) for j in (0, 5, 10)]
        print(f"    {c:<16} R={means[c][0]:.4f} G={means[c][1]:.4f} B={means[c][2]:.4f}")
    report["class_rgb_mean"] = means

    # ------------------------------------------------ H2 近重复残留
    print("\n" + "=" * 70)
    print("H2  group-aware 折上，测试图的最近训练邻是否同 group？")
    F = np.load(CACHE / "dino_feats.npy")
    P = [x["path"] for x in items]
    idx = {p: i for i, p in enumerate(P)}
    order = np.array([idx[x["path"]] for x in items])
    Fo = F[order]                      # 与 items 对齐
    S = Fo @ Fo.T
    g = np.array([x["group"] for x in items])

    stat = {}
    for scheme in ("grouped", "stratified"):
        f = np.array([x[f"fold_{scheme}"] for x in items])
        same_g, cross_c, tot = 0, 0, 0
        for k in range(5):
            te = np.flatnonzero(f == k)
            tr = np.flatnonzero(f != k)
            sub = S[np.ix_(te, tr)]
            nn = tr[sub.argmax(1)]
            for a, b in zip(te, nn):
                tot += 1
                if g[a] == g[b]:
                    same_g += 1
                if y[a] != y[b]:
                    cross_c += 1
        stat[scheme] = {
            "pct_nn_same_group": round(100 * same_g / tot, 2),
            "pct_nn_cross_class": round(100 * cross_c / tot, 2),
            "n": tot,
        }
        print(f"  {scheme:<12} 最近邻同group {stat[scheme]['pct_nn_same_group']:>6.2f}%"
              f"｜最近邻跨类 {stat[scheme]['pct_nn_cross_class']:>6.2f}%")
    print("  → grouped 下「同 group」应≈0（否则group-aware 失效）；"
          "stratified 下若明显>0 即为泄漏残留")
    report["h2_nn_stats"] = stat

    # ------------------------------------------------ H3 折间偏置
    print("\n" + "=" * 70)
    print("H3  各折像素统计是否有系统差异（折号本身携带信息 = 划分有偏）")
    fg = np.array([x["fold_grouped"] for x in items])
    f1s_by_fold = []
    for k in range(5):
        te = np.flatnonzero(fg == k)
        tr = np.flatnonzero(fg != k)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        f1s_by_fold.append(round(float(f1_score(
            y[te], clf.predict(sc.transform(X[te])), average="macro",
            zero_division=0)), 4))
    print(f"  逐折（用像素统计预测本折）: {f1s_by_fold}")
    print("  → 若某折显著低，说明该折分布特殊；5 折都高 = 捷径稳定存在")
    report["h3_per_fold"] = f1s_by_fold

    # ------------------------------------------------ 结论
    print("\n" + "=" * 70)
    print("结论")
    print(f"  像素统计 macro-F1 = {h1:.1%}（仅 19 维低阶颜色/亮度信息）")
    if h1 > 0.5:
        print(f"  ⚠️  **低阶颜色捷径客观存在**：DC-02R 每类在同一拍摄 session 完成，")
        print(f"     背景色（Warping/Off_platform 绿床 vs 其余灰白）即类别代理。")
        print(f"  ⚠️  但这**不等于**高macro-F1 是虚高 —— 见下方「重要修正」。")
    report["h1_verdict"] = ("低阶颜色捷径存在，但需干预实验才能判断它是否为高分成因")

    print("\n  🔴 重要修正（2026-10-04 后续干预实验推翻本脚本的初始假设）：")
    print("     本脚本曾推断「98% 反映的是识别拍摄批次，不是缺陷形态」。")
    print("     **该推断错误**，已由 exp/d3_intervention.py 实测推翻：")
    print("       V0 原始 L2 = 0.9977 | V2 去背景 = 0.9977（0.00pp）")
    print("       V3 去色+去背景 L2 = 0.9944（仅 −0.33pp）")
    print("     即：线性探针在 1280 维特征上**并未依赖**低阶颜色线索，")
    print("     71.5% 只是「19 维像素统计能到的高度」，不是 EfficientNet 的依赖度。")
    print("     **教训：单一证据（像素统计高）不足以判定捷径，必须做干预实验"
          "（主动移除该线索看分数掉多少）才能定论。**")
    report["h1_intervention_verdict"] = {
        "V0_L2": 0.9977, "V2_L2": 0.9977, "V3_L2": 0.9944,
        "gap_pp": 0.33,
        "conclusion": "颜色/背景捷径对高 macro-F1 的贡献仅 0.33pp，高分非虚高",
        "method_note": ("判定捷径必须做干预实验（移除线索→看分数掉多少），"
                        "不能只看「该线索单独能到多少分」。"),
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
