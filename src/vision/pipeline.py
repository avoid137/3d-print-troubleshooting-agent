"""P1 · 端到端诊断管线：缺陷图 → 5 类 → KB 条目（含易混淆与冲突说明）。

这是 P1 的**交付面**：上游是冻结骨干 + 线性探针，下游是 kb_v2 知识库。
中间这一层把「分类结果」翻译成「可执行的排查建议」。

🔴 三条设计纪律：

1. **拒答（reject）是一等公民**，不是可选开关。
   分类器有 5 类，但知识库对某些现象的处置是**互斥**的
   （如 Spaghetti 要查挤出/堵塞，脱离平台只需查粘床）。
   给一个低置信度的确定答案，比说「我不确定」更危险——
   因为用户会照着错误建议去调参数。
   故：max proba < 阈值 → 拒答，但**仍输出 top-2 候选**供人工判断。

   ✅ 阈值**已用折外预测标定**（不是拍脑袋，也不是在部署件上自证）：
      错误样本最高概率 mean 0.6910/ min 0.4919
      正确样本最高概率 mean 0.9957 / min 0.5576
      → 置信度能区分对错，拒答**确实**能捞出 5/6 的错误（th=0.90）。
   ⚠️ 但反过来说：**高置信度也不等于正确**。th=0.90 下仍有 1/6 错误
      带着 ≥0.90 的置信度溜过去；th=0.999 才能全拦住，代价是覆盖率掉到 78.5%。
      且**只有 6 个错误样本**，阈值的统计基础极薄。
      结论：拒答是**减少**错误暴露的手段，不是消除错误的手段。

2. **易混淆项必须一起输出，不能只给主映射。**
   下游若只拿到 KB101 而看不到 KB102/103，会漏掉「喷嘴堵塞」这条根因；
   只拿到 KB009 而看不到 KB049，会把层间粘合问题误判成风扇/挤出温度问题。
   故 confusable_kb 与 conflict_reason 一并透传。

3. **映射是纯查表，与模型解耦。**
   改 kb_mapping 里的映射**不需要重训模型**（推理侧只用到 classes 顺序）。
   反过来，模型换了也不影响映射。二者分别版本化。

    🔴 但要注意：2026-10-05 看图定稿后，**Off_platform 的实际现象是 Spaghetti**
    （类名与现象不符，为对齐数据集标签未改名）。
    这意味着本管线对Off_platform 输出的是「挤出失败」类建议——
    语义正确，但类名字面会让人意外。`kb_details` 里已带`zh` 中文名兜底。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.vision.infer import (  # noqa: E402
    CLASSES, Prediction, load_deployed_model, predict_probs,
)

#: 拒答阈值。**已用折外预测标定**（exp/d3_reject_calibration.py），非拍脑袋。
#:
#: 实测（group-aware 折外，n=1912，错 6 张）：
#:     正确样本最高概率 mean 0.9957 / min 0.5576
#:     错误样本最高概率 mean 0.6910 / min 0.4919
#:     → 置信度能区分对错（差0.30），拒答机制有效
#:权衡（覆盖率 vs 错误拦截率）：
#:     th=0.50回答 99.9%  拦下错误 33.3%  回答中正确率 99.79%
#:     th=0.90  回答 98.6%  拦下错误 83.3%  回答中正确率 99.95%
#:     th=0.999 回答 78.5%  拦下错误100%   回答中正确率 100%
#:
#: 默认取 0.90：拦下 5/6错误（83.3%），覆盖率仍 98.6%。
#: ⚠️ **只有 6 个错误样本**，这个阈值是「起点」不是「定论」。
#: 业务上「误伤一个好件」与「放过一个坏件」的代价不同，
#: 拿到真实场景数据后必须重新标定。
DEFAULT_REJECT = 0.90


def diagnose(paths, reject_threshold: float = DEFAULT_REJECT,
             device: str = "cuda", with_confusable: bool = True,
             model_dir=None) -> list:
    """对一批图片跑完整链路，返回每张的 Prediction。"""
    from src.vision.kb_mapping import build_mapping, load_kb

    paths = [str(p) for p in paths]
    if not paths:
        return []
    scaler, clf, model, meta = load_deployed_model(model_dir, device=device)
    proba = predict_probs(paths, scaler, clf, model=model, device=device)

    kb = load_kb()
    mapping = build_mapping(kb)

    out = []
    for i, p in enumerate(paths):
        row = proba[i]
        order = np.argsort(row)[::-1]
        top1 = int(order[0])
        cls1 = CLASSES[top1]
        p1 = float(row[top1])
        top2 = [(CLASSES[int(j)], float(row[int(j)])) for j in order[:2]]

        mp = mapping[cls1]
        kb_ids = [k.kb_id for k in mp.primary + mp.secondary]
        if with_confusable:
            kb_ids += [k.kb_id for k in mp.confusable]

        details = []
        for k in mp.primary + mp.secondary + mp.confusable:
            details.append({
                "kb_id": k.kb_id, "title": k.title, "role": k.role,
                "match": k.match, "cause_count": k.cause_count, "note": k.note,
            })

        rejected = p1 < reject_threshold
        reason = ""
        if rejected:
            gap = p1 - top2[1][1] if len(top2) > 1 else p1
            reason = (f"最高类概率 {p1:.2f} < 阈值 {reject_threshold:.2f}"
                      + (f"，且与第二名仅差 {gap:.2f}" if gap < 0.15 else "")
                      + " → 不给确定建议")

        out.append(Prediction(
            pred_class=cls1, proba=p1, top2=top2,
            kb_ids=list(dict.fromkeys(kb_ids)),
            kb_details=details,
            rejected=rejected, reject_reason=reason, source_path=p))
    return out


def format_report(preds: list) -> str:
    """人可读的报告（给 P0 的问答层 / 演示用）。"""
    L = []
    for i, p in enumerate(preds, 1):
        L.append(f"\n【{i}】{Path(p.source_path).name if p.source_path else ''}")
        L.append(f"   预测类别: {p.pred_class}  概率 {p.proba:.4f}")
        L.append("   Top-2: " + "  |  ".join(f"{c} {q:.4f}" for c, q in p.top2))
        if p.rejected:
            L.append(f"   ⚠️拒答: {p.reject_reason}")
            L.append("   → 仍给出 top-2 供人工判断；建议补拍/换角度后重试")
        prim = [d for d in p.kb_details if d["role"] == "primary"]
        sec = [d for d in p.kb_details if d["role"] == "secondary"]
        cnf = [d for d in p.kb_details if d["role"] == "confusable"]
        if prim:
            L.append("   主映射: " + "; ".join(
                f"{d['kb_id']} {d['title']}（{d['cause_count']} 根因）" for d in prim))
        if sec:
            L.append("   次要:   " + "; ".join(f"{d['kb_id']} {d['title']}" for d in sec))
        if cnf:
            L.append("   ⚠️易混淆（**不是同一现象**）: " + "; ".join(
                f"{d['kb_id']} {d['title']}" for d in cnf))
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reject-threshold", type=float, default=DEFAULT_REJECT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    preds = diagnose(a.images, reject_threshold=a.reject_threshold,
                     device=a.device)
    print(format_report(preds))

    if a.out:
        Path(a.out).write_text(json.dumps(
            [p.to_dict() for p in preds], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n[产出] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
