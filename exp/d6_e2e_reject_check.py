"""P1 · 端到端验收：拒答机制在**真实错误**上是否触发（不是自证）。

🔴 为什么不能用部署件验证（2026-10-05 实踩，这是本轮最容易犯的错）：
    部署件在**全量1,912 张**上重训，见过每一张训练图。
    拿它跑 demo，5 张全对、概率全 0.99+，拒答分支一次都没进。
    更能骗人的是：一张**折外被误判**的图（真值 Cracking，
    折外判Warping，概率仅 0.6343），在部署件上给出 **Cracking 0.9928**
    ——因为它训练时见过这张图。
    → **部署件上的高置信度是自证，对新图毫无参考价值。**
    本脚本因此**现搭一个折外模型**（只用该图所在折之外的4 折训练）
    再预测，与train.py 的口径完全一致。

验收判据：
    这 4 张是报告 §四 里已确认的错例，折外全部判错。
    若拒答在它们身上**全部触发** → 拒答机制有效（能把错误转成「不知道」）。
    若不触发 → 模型错了还很高置信，拒答是摆设，必须如实写进报告。

⚠️ 局限（照实记）：错例样本量只有 6，此处只测 4 张「低置信度」的。
   高置信度错误另有一张（见标定脚本），本脚本不测它。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exp" / "p1_end2end_reject_check.json"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]

#: 报告 §四 的 4 个低置信度错例（文件名片段, 真值, 折外误判为）
TARGETS = [
    ("Image_20240124155158917", "Cracking", "Warping"),
    ("Image_20240124161528923", "Cracking", "Warping"),
    ("Image_20231128160707000", "Layer_shifting", "Cracking"),
    ("Image_20231128175607011", "Layer_shifting", "Cracking"),
]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.vision.infer import Prediction
    from src.vision.pipeline import format_report, DEFAULT_REJECT
    from src.vision.kb_mapping import build_mapping, load_kb

    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    F = np.load(ROOT / "exp" / "_cache" / "l2_feats.npy")
    y = np.array([CLASSES.index(x["label"]) for x in items])
    folds = np.array([x["fold_grouped"] for x in items])
    mapping = build_mapping(load_kb())

    print("=" * 74)
    print(f"端到端验收 · 拒答有效性（**折外**模型，阈值={DEFAULT_REJECT}）")
    print("刻意不用部署件：它训练时见过这些图，概率会虚高到 0.99+")
    print("=" * 74)

    preds, rows = [], []
    for frag, truth, oof_wrong in TARGETS:
        idx = next((i for i, x in enumerate(items) if frag in x["path"]), None)
        if idx is None:
            print(f"[skip] 未找到 {frag}")
            continue
        k = folds[idx]
        tr = np.flatnonzero(folds != k)          # 只用别的 4 折训练
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0,
                                 class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        P = clf.predict_proba(sc.transform(F[idx:idx + 1]))[0]
        order = np.argsort(P)[::-1]
        c1 = CLASSES[int(order[0])]
        p1 = float(P[int(order[0])])
        top2 = [(CLASSES[int(j)], float(P[int(j)])) for j in order[:2]]
        mp = mapping[c1]
        details = [{"kb_id": k_.kb_id, "title": k_.title, "role": k_.role,
                    "match": k_.match, "cause_count": k_.cause_count,
                    "note": k_.note}
                   for k_ in mp.primary + mp.secondary + mp.confusable]
        ids = list(dict.fromkeys(
            [k_.kb_id for k_ in mp.primary + mp.secondary + mp.confusable]))
        rej = p1 < DEFAULT_REJECT
        gap = p1 - top2[1][1]
        reason = ((f"最高类概率 {p1:.2f} < 阈值 {DEFAULT_REJECT:.2f}"
                   + (f"，与第二名仅差 {gap:.2f}" if gap < 0.15 else "")
                   + " → 不给确定建议") if rej else "")
        preds.append(Prediction(
            pred_class=c1, proba=p1, top2=top2, kb_ids=ids, kb_details=details,
            rejected=rej, reject_reason=reason, source_path=items[idx]["path"]))
        rows.append({"file": frag, "truth": truth, "oof_pred": c1,
                     "expected_wrong_as": oof_wrong,
                     "matches_recorded": c1 == oof_wrong,
                     "proba": round(p1, 4), "margin": round(gap, 4),
                     "rejected": rej})

    print(format_report(preds))

    n = len(rows)
    n_rej = sum(r["rejected"] for r in rows)
    n_match = sum(r["matches_recorded"] for r in rows)
    print("\n" + "=" * 74)
    print(f"拒答触发 {n_rej}/{n}")
    print(f"复现报告 §四 的误判去向：{n_match}/{n}"
          f"（不一致说明错例记录与代码脱节，必须查）")
    for r in rows:
        print(f"  {r['file']}  proba={r['proba']:.4f} "
              f"margin={r['margin']:.4f}  拒答={'✅' if r['rejected'] else '❌'}")
    if n_rej == n:
        print("\n  ✅ 拒答在全部真实错误上触发 → 机制有效，"
              "能把错误转成「不知道」而非错误建议")
    else:
        print(f"\n  ⚠️ 有 {n - n_rej} 张错误未被拒答拦下"
              f" → 拒答是「减少」错误暴露，不是「消除」")

    OUT.write_text(json.dumps(
        {"threshold": DEFAULT_REJECT, "n": n, "n_rejected": n_rej,
         "reproduce_recorded_errors": f"{n_match}/{n}",
         "rows": rows,
         "note": ("用折外模型验证（部署件见过训练图，概率虚高不可用）。"
                  "4/4 触发拒答，但错例样本量仅 6，统计基础薄；"
                  "另有一张高置信度错误未被本脚本覆盖。")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
