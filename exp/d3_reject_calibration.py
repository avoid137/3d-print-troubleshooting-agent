"""P1 · 拒答机制有效吗？用**折外**预测检验（不是部署件上的自证）。

🔴 为什么必须单独做这个检验（2026-10-05 实踩）：
    我在部署件（全量重训）上跑 5 张 demo，概率全是 0.99+，
    拒答分支**一次都没触发**。更糟的是：一张已知错例
    （真值 Cracking / 折外被误判 Warping 的 Image_20240124155158917）
    在部署件上给出 **Cracking 0.9928** —— 因为它见过这张训练图。
    → **部署件上的高置信度完全不能证明拒答阈值合理**，那是自证。

正确做法：在 **group-aware 折外**预测上检验
    「模型错的时候，置信度是否真的更低？」
      · 若错误样本的最高概率显著低于正确样本 → 拒答阈值能捞出部分错误
      · 若错误样本也是 0.99→ 阈值没用，必须如实说「模型错了但很自信」

🔴 阈值选择纪律：
    阈值不能照着部署件的经验拍脑袋（0.60 是拍的）。
    本脚本**从折外预测里扫阈值**，给出「覆盖率 vs 错误拦截率」的权衡曲线，
    让人看到代价，而不是只报一个好看的数字。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exp" / "p1_d3_reject_calibration.json"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]


def oof_predict():
    """group-aware 折外概率矩阵 [N, 5]。每张图都由没见过它的模型预测。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    F = np.load(ROOT / "exp" / "_cache" / "l2_feats.npy")
    y = np.array([CLASSES.index(x["label"]) for x in items])
    folds = np.array([x["fold_grouped"] for x in items])
    P = np.zeros((len(items), len(CLASSES)))
    for k in sorted(set(folds.tolist())):
        te = np.flatnonzero(folds == k)
        tr = np.flatnonzero(folds != k)
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0,
                                 class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        P[te] = clf.predict_proba(sc.transform(F[te]))
    return P, y, items


def main():
    print("=" * 72)
    print("拒答阈值标定：折外预测（非部署件）")
    P, y, items = oof_predict()
    top1 = P.argmax(1)
    maxp = P.max(1)
    correct = (top1 == y)
    print(f"n={len(y)}  折外准确率 = {correct.mean():.4%}  错{(~correct).sum()} 张")

    print("\n--- 正确 vs 错误 的最高概率分布 ---")
    for name, m in [("正确", correct), ("错误", ~correct)]:
        v = maxp[m]
        print(f"  {name}: n={m.sum():>4d}  mean {v.mean():.4f}  "
              f"median {np.median(v):.4f}  p10 {np.percentile(v,10):.4f}  "
              f"min {v.min():.4f}")
    print(f"  ⚠️ 错误的最低最高概率 = {maxp[~correct].min():.4f}"
          f"  ← 若接近 1.0，说明模型错了也很自信，阈值救不了")

    print("\n--- 阈值扫描：覆盖率 / 错误拦截率 ---")
    rows = []
    n_err = int((~correct).sum())
    for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]:
        answered = maxp >= th
        n_ans = int(answered.sum())
        cov = n_ans / len(y)
        caught = int((~correct & ~answered).sum())      # 被拒答挡住的错误
        block = caught / n_err if n_err else 0.0
        # 答对率（回答了的里面有多少是对的）
        prec = float(correct[answered].mean()) if n_ans else 0.0
        rows.append({"threshold": th, "coverage": round(cov, 4),
                     "answer_rate": round(cov, 4),
                     "errors_caught": caught, "err_block_rate": round(block, 4),
                     "precision_when_answered": round(prec, 4)})
        print(f"  th={th:<7} 回答 {cov*100:5.1f}%  "
              f"拦下错误 {caught}/{n_err} ({block*100:4.1f}%)  "
              f"回答中正确率 {prec*100:5.2f}%")

    # 关键结论判定
    lo = maxp[~correct].min()
    hi_correct = maxp[correct].min()
    if lo < 0.5:
        print(f"\n  ✅ 存在低置信度的错误样本（最低 {lo:.4f}）"
              f" → 拒答机制**能**捞出至少一部分错误")
    else:
        print(f"\n  ❌ 错误样本的最高概率最低也有 {lo:.4f}"
              f" → **模型错了也很自信**，拒答几乎救不了")
    print(f"  参考：正确样本里概率最低的一张也有 {hi_correct:.4f}"
          f" → 高置信度仍不等于正确")
    print(f"  ⚠️ 本项目只有 {n_err} 个错误样本，阈值标定样本量极小，"
          f"**该阈值只能当起点，不能当定论**")

    OUT.write_text(json.dumps(
        {"n": len(y), "oof_accuracy": round(float(correct.mean()), 5),
         "n_errors": n_err,
         "conf_correct": {"mean": round(float(maxp[correct].mean()), 5),
                          "median": round(float(np.median(maxp[correct])), 5),
                          "min": round(float(maxp[correct].min()), 5)},
         "conf_error": {"mean": round(float(maxp[~correct].mean()), 5),
                        "median": round(float(np.median(maxp[~correct])), 5),
                        "min": round(float(maxp[~correct].min()), 5)},
         "threshold_scan": rows,
         "note": ("阈值从**折外**预测标定，不是在部署件上自证。"
                  "仅 6 个错误样本 → 样本量极小，阈值只当起点。"
                  "pipeline.py 的 0.60 是起点，实际应按业务成本调。")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
