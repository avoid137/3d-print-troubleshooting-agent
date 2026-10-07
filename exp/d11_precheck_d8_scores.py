"""D11 预检 · 用已落盘的 d8 域外样本，离线判断「置信度类 OOD 分数」是否可用。

背景
----
P1 视觉的核心问题不是分类精度（同域 macro-F1 99.73%），而是**跨域塌缩**
（DC-04 仅 6.25%，低于 5 类随机 20%）。定下的处置是「域受控 + 域外安全阀」：
不追求跨任意域分类准确，而是**让系统知道输入已漂出训练域**，从而降级/报警。

安全阀的第一候选是最便宜的「置信度类分数」（max-softmax / margin）——
因为它不需要额外特征，推理时就有。但 d8 的阈值扫描已暗示它可能**没用**：
阈值 0.9 时域外错误只抓住 24.8%，却把 75% 的正常件也拒了。

本脚本零成本离线回答一个问题：

    「域外样本的 max-softmax 分布，与域内差多少？能不能分开？」

做法
----
· 域外：直接读 exp/d8_external_result.json 的 plan（1200 条逐样本 proba/top2）
· 域内：读 exp/p1_d3_reject_calibration.json 的 threshold_scan
        （域内 1912 条 OOF 的「阈值 → coverage=P(proba>=th)」曲线）
· 两边拼出 ROC，梯形法估 AUROC；并给工作点表（域外检出率 vs 域内误报率）

⚠️ 局限（必须随结论一起引用）
· 域内是**汇总的 coverage 曲线**，不是逐样本，阈值点稀疏（0.5~0.9999）→ AUROC 是**近似**。
· d8 是 5 类模型（无 No_error），所以「域外准确率」不可用，本脚本只谈 **OOD 可分性**。
· 真正的定论必须在 4060 上用逐样本特征重算（见 exp/d11_ood_probe.py）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D8 = ROOT / "exp" / "d8_external_result.json"
REJECT_CAL = ROOT / "exp" / "p1_d3_reject_calibration.json"
OUT = ROOT / "exp" / "d11_precheck_result.json"


def quantiles(xs: list[float], qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)

    def q(p):
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {f"p{int(p*100):02d}": round(q(p), 5) for p in qs}


def main() -> None:
    # ---------------- 域外（DC-04，跨域） ----------------
    d8 = json.load(open(D8, encoding="utf-8"))
    plan = d8["plan"]
    ood = {
        "n": len(plan),
        "proba": [float(s["proba"]) for s in plan],
        "margin": [],
        "by_view": {},
        "by_cat": {},
    }
    for s in plan:
        t2 = s.get("top2") or []
        if len(t2) >= 2:
            ood["margin"].append(float(t2[0][1]) - float(t2[1][1]))
        ood["by_view"].setdefault(s["view"], []).append(float(s["proba"]))
        ood["by_cat"].setdefault(s["cat"], []).append(float(s["proba"]))

    # ---------------- 域内（DC-02R，同域）的 coverage 曲线 ----------------
    cal = json.load(open(REJECT_CAL, encoding="utf-8"))
    scan = cal["threshold_scan"]  # [{threshold, coverage(=P(proba>=th)), ...}, ...]
    id_pts = sorted((float(r["threshold"]), float(r["coverage"])) for r in scan)
    # coverage(th) = P(域内 proba >= th) → 域内 CDF = 1 - coverage
    n_id = cal["n"]

    def id_cdf(th: float) -> float:
        """域内 P(proba < th)，由离散 coverage 曲线线性插值。"""
        if th <= id_pts[0][0]:
            return 1.0 - id_pts[0][1]
        if th >= id_pts[-1][0]:
            return 1.0 - id_pts[-1][1]
        for (t0, c0), (t1, c1) in zip(id_pts, id_pts[1:]):
            if t0 <= th <= t1:
                w = 0.0 if t1 == t0 else (th - t0) / (t1 - t0)
                return 1.0 - (c0 * (1 - w) + c1 * w)
        return 1.0 - id_pts[-1][1]

    # ---------------- 拼 ROC：阈值为「低于 th 判 OOD」 ----------------
    grid = [i / 1000 for i in range(0, 1001)]
    ood_sorted = sorted(ood["proba"])
    n_ood = len(ood_sorted)

    def ood_tpr(th: float) -> float:
        """域外 TPR = P(域外 proba < th)，bisect 计数。"""
        import bisect
        return bisect.bisect_left(ood_sorted, th) / n_ood

    roc = []
    for th in grid:
        tpr = ood_tpr(th)          # 域外被正确识别为 OOD 的比例
        fpr = id_cdf(th)           # 域内被误判为 OOD 的比例
        roc.append((fpr, tpr))

    # 梯形法 AUROC（对 fpr 排序）
    roc_sorted = sorted(roc)
    auroc = 0.0
    for (f0, t0), (f1, t1) in zip(roc_sorted, roc_sorted[1:]):
        auroc += (f1 - f0) * (t0 + t1) / 2

    # ---------------- 关键工作点 ----------------
    work = []
    for fpr_target in (0.01, 0.05, 0.10, 0.20):
        # 在不超过该 FPR 的前提下，取最大的 TPR
        best = max((t for f, t in roc if f <= fpr_target), default=0.0)
        work.append({"fpr_id": fpr_target, "tpr_ood": round(best, 4)})

    # ---------------- 报告 ----------------
    res = {
        "_note": "置信度类 OOD 分数可用性预检（域外=d8 DC-04 逐样本；域内=D3 coverage 汇总曲线）",
        "ood_n": n_ood,
        "id_n": n_id,
        "ood_maxsoftmax_quantiles": quantiles(ood["proba"]),
        "ood_margin_quantiles": quantiles(ood["margin"]),
        "ood_by_view_quantiles": {k: quantiles(v) for k, v in ood["by_view"].items()},
        "ood_by_cat_quantiles": {k: quantiles(v) for k, v in ood["by_cat"].items()},
        "id_maxsoftmax": {
            "quantiles_from_summary": {
                "conf_correct_median": cal["conf_correct"]["median"],
                "conf_correct_mean": cal["conf_correct"]["mean"],
                "conf_error_median": cal["conf_error"]["median"],
                "conf_error_mean": cal["conf_error"]["mean"],
            },
            "cdf_at": {str(th): round(id_cdf(th), 5) for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)},
        },
        "rate_ge_0.9": {
            "ood": round(sum(1 for p in ood["proba"] if p >= 0.9) / n_ood, 4),
            "id": round(1 - id_cdf(0.9), 4),
        },
        "auroc_maxsoftmax_approx": round(auroc, 4),
        "working_points": work,
        "_caveat": "域内为汇总 coverage 曲线（非逐样本），AUROC 为近似；定论须在 4060 用逐样本特征重算。",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print("=" * 68)
    print("D11 预检 · 置信度类 OOD 分数可用性（域外 d8 DC-04 vs 域内 D3）")
    print("=" * 68)
    print(f"域外样本 n={n_ood}  域内样本 n={n_id}")
    print("\n[域外 max-softmax 分位数]")
    for k, v in res["ood_maxsoftmax_quantiles"].items():
        print(f"   {k}: {v}")
    print("\n[域外 margin(top1-top2) 分位数]")
    for k, v in res["ood_margin_quantiles"].items():
        print(f"   {k}: {v}")
    print("\n[按 DC-04 类别]")
    for k, v in res["ood_by_cat_quantiles"].items():
        print(f"   {k:26s} {v}")
    print(f"\n[P(proba>=0.9)]  域外={res['rate_ge_0.9']['ood']:.1%}  域内={res['rate_ge_0.9']['id']:.1%}")
    print(f"\n[近似 AUROC(max-softmax 做 OOD)] = {res['auroc_maxsoftmax_approx']}")
    print("   (0.5=完全无区分度；>0.8 才算可用)")
    print("\n[工作点：限定域内误报 FPR 下的域外检出率]")
    for w in work:
        print(f"   域内误报<={w['fpr_id']:.0%}  →  域外检出={w['tpr_ood']:.1%}")
    print(f"\n已落盘: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
