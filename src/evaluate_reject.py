"""拒答评测：LLM 判官 vs 余弦阈值基线（全量 200 条）。

指标定义（关键：分母统一为各类总数，才能跨方案对比）
--------------------------------------------------
40 条库外负例（out_of_scope）：
  正确拒答率 = 拒答数 / 40          ← 越高越好
  越界回答率 = 被误答数 / 40        ← 越低越好（本该拒答却硬答）

160 条库内正例：
  有效回答率 = (未拒答 且 命中 gold) / 160   ← **端到端体感，主指标**
  误答率     = (未拒答 且 未命中) / 160      ← 最危险：自信地答错
  误拒率     = 拒答数 / 160                  ← 把该答的推掉了

拒答器本身的分类质量：拒答 Precision / Recall / F1（视「拒答」为正类）

⚠️ 两个已踩过的度量坑（勿重犯）
--------------------------------------------------
坑 1「条件口径」：分母写成「被放行的正例数」→ 拒答越激进分母越小、数字越虚高。
    阈值法靠拒掉 45% 正例把该口径刷到 81.82%，是幸存者偏差。
坑 2「纯检索口径」：分子用「检索命中」而不扣除被拒答的条目 → 该指标只反映检索，
    与拒答机制无关，任何拒答方案都得到同一个数（实测两者都 = 136/160），无区分度。
两者都错。正解：分子 = 「未拒答 且 命中」，分母 = **全部正例**。

对比基线：纯余弦阈值法（实测最优阈值 0.6865）
"""
import json
import sys

from config import EVAL_SET_PATH, RECALL_K, PROJECT_ROOT
from hybrid_retriever import HybridRetriever
from rejector import judge_many, threshold_baseline, JUDGE_MODEL, PROMPT_VERSION

OUT_PATH = PROJECT_ROOT / "data" / "reject_eval_result.json"


def summarize(rows, label):
    """rows: [{category, gold, retrieved_kb, rejected}]

    gold 为空列表 ⇒ 库外负例；非空 ⇒ 库内正例。
    """
    neg = [r for r in rows if not r["gold"]]
    pos = [r for r in rows if r["gold"]]

    def is_hit(r):
        return any(k in r["gold"] for k in r["retrieved_kb"])

    # ---- 负例：40 条 ----（拒答视为正类）
    neg_rej = sum(1 for r in neg if r["rejected"])
    neg_answered = len(neg) - neg_rej

    # ---- 正例：160 条 ----
    passed = [r for r in pos if not r["rejected"]]
    pos_rej = len(pos) - len(passed)
    answered_hit = sum(1 for r in passed if is_hit(r))
    answered_miss = len(passed) - answered_hit

    # 拒答器分类质量（拒答 = 正类）
    rej_recall = neg_rej / len(neg) if neg else 0.0
    rej_prec = neg_rej / (neg_rej + pos_rej) if (neg_rej + pos_rej) else 0.0
    f1 = (2 * rej_prec * rej_recall / (rej_prec + rej_recall)
          if (rej_prec + rej_recall) else 0.0)

    n_pos, n_neg = len(pos), len(neg)
    out = {
        "label": label,
        # 负例（库外）
        "neg_total": n_neg,
        "neg_rejected": neg_rej,
        "rejection_recall": round(rej_recall, 4),      # 正确拒答率
        "neg_answered": neg_answered,
        "neg_answered_rate": round(neg_answered / n_neg, 4) if n_neg else 0.0,  # 越界回答率
        # 正例（库内）
        "pos_total": n_pos,
        "pos_rejected": pos_rej,
        "false_reject_rate": round(pos_rej / n_pos, 4) if n_pos else 0.0,  # 误拒率
        "answered_hit": answered_hit,
        "answered_miss": answered_miss,
        "answer_hit_rate": round(answered_hit / n_pos, 4) if n_pos else 0.0,   # ← 主指标
        "answer_miss_rate": round(answered_miss / n_pos, 4) if n_pos else 0.0, # 误答率
        # 拒答器本身的分类质量
        "reject_precision": round(rej_prec, 4),
        "reject_f1": round(f1, 4),
    }
    out["_check"] = (out["answer_hit_rate"] + out["answer_miss_rate"]
                     + out["false_reject_rate"])

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print("=" * 70)
    print(f"  【库内 {n_pos} 条正例 · 端到端】")
    print(f"    有效回答率(答对) : {answered_hit}/{n_pos} = "
          f"{out['answer_hit_rate']:.4f}  ← 主指标")
    print(f"    误答率(自信答错) : {answered_miss}/{n_pos} = "
          f"{out['answer_miss_rate']:.4f}")
    print(f"    误拒率(该答没答) : {pos_rej}/{n_pos} = "
          f"{out['false_reject_rate']:.4f}")
    print(f"  【库外 {n_neg} 条负例】")
    print(f"    正确拒答率       : {neg_rej}/{n_neg} = {out['rejection_recall']:.4f}")
    print(f"    越界回答率       : {neg_answered}/{n_neg} = "
          f"{out['neg_answered_rate']:.4f}")
    print(f"  【拒答器分类质量】Precision {rej_prec:.4f} / F1 {f1:.4f}")
    print(f"  【一致性校验】三项之和 = {out['_check']:.4f}（应为 1.0000）")
    return out


def main():
    # --reuse：从已存明细重算指标，不重新检索 / 不调 API。
    # 用途：指标定义调整后要重新汇总时，避免重复消耗 embedding 调用。
    if "--reuse" in sys.argv:
        if not OUT_PATH.exists():
            print(f"❌ 未找到明细 {OUT_PATH}，无法 --reuse；请直接全量运行。")
            return
        saved = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        rows_llm = saved["llm_rows"]
        rows_th = saved.get("threshold_rows")
        if not rows_th:
            print("❌ 明细缺少 threshold_rows（旧版本生成），请全量重跑一次。")
            return
        print(f"[--reuse] 从 {OUT_PATH.name} 读取 "
              f"{len(rows_llm)} + {len(rows_th)} 条明细，仅重算指标。")
    else:
        ev = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
        r = HybridRetriever()
        print(f"检索 {len(ev)} 条 ...  （索引后端: {r.embed_method}）")
        pairs = []
        for e in ev:
            res = r.retrieve(e["query"], top_k=RECALL_K)
            pairs.append((e, res))

        print(f"判官判断 {len(ev)} 条 ...")
        verdicts = judge_many([(e["query"], res) for e, res in pairs])

        # ---- 方案 A：LLM 判官 ----
        rows_llm = []
        for (e, res), v in zip(pairs, verdicts):
            rows_llm.append({
                "id": e["id"],
                "category": e["category"],
                "query": e["query"],
                "gold": e["gold"],
                "retrieved_kb": [x["kb_id"] for x in res],
                "top_cosine": res[0]["cosine"] if res else -1.0,
                "rejected": not v["answerable"],
                "reason": v.get("reason", ""),
            })

        # ---- 方案 B：余弦阈值基线 ----
        rows_th = []
        for e, res in pairs:
            rows_th.append({
                "id": e["id"],
                "category": e["category"],
                "query": e["query"],
                "gold": e["gold"],
                "retrieved_kb": [x["kb_id"] for x in res],
                "top_cosine": res[0]["cosine"] if res else -1.0,
                "rejected": not threshold_baseline(res),
            })

    r_llm = summarize(rows_llm, f"方案 A · LLM 判官（{JUDGE_MODEL}, prompt {PROMPT_VERSION}）")
    r_th = summarize(rows_th, "方案 B · 余弦阈值基线（阈值 0.6865）")

    print(f"\n{'=' * 70}")
    print("对比结论")
    print("=" * 70)
    print(f"  【端到端·主指标】有效回答率 : 判官 {r_llm['answer_hit_rate']:.4f}"
          f"  vs  阈值 {r_th['answer_hit_rate']:.4f}")
    print(f"                   误答率     : 判官 {r_llm['answer_miss_rate']:.4f}"
          f"  vs  阈值 {r_th['answer_miss_rate']:.4f}")
    print(f"                   误拒率     : 判官 {r_llm['false_reject_rate']:.4f}"
          f"  vs  阈值 {r_th['false_reject_rate']:.4f}")
    print(f"  【库外】正确拒答率          : 判官 {r_llm['rejection_recall']:.4f}"
          f"  vs  阈值 {r_th['rejection_recall']:.4f}")
    print(f"          越界回答率          : 判官 {r_llm['neg_answered_rate']:.4f}"
          f"  vs  阈值 {r_th['neg_answered_rate']:.4f}")
    print(f"  【拒答器】F1                : 判官 {r_llm['reject_f1']:.4f}"
          f"  vs  阈值 {r_th['reject_f1']:.4f}")
    gain = r_llm["answer_hit_rate"] - r_th["answer_hit_rate"]
    print(f"\n  → 判官把有效回答率拉高 {gain:+.4f}（+{gain * 100:.2f}pp），"
          f"代价是误答率 {r_th['answer_miss_rate']:.4f} → {r_llm['answer_miss_rate']:.4f}。")
    print("     阈值法并未答得更准，它只是「少答」——把 45% 的有效问题也推掉了，")
    print("     所以误答率低是『少犯错』而非『答得好』。")

    # 判官误拒的正例明细（供人工复核）
    bad = [r for r in rows_llm if r["gold"] and r["rejected"]]
    if bad:
        print(f"\n判官误拒的正例（{len(bad)} 条，需人工复核）：")
        for r in bad:
            print(f"  - [{r['category']}] {r['query'][:45]}")
            print(f"      gold={r['gold']} cos={r['top_cosine']:.3f} | {r['reason'][:60]}")

    OUT_PATH.write_text(json.dumps({
        "llm_judge": r_llm,
        "threshold_baseline": r_th,
        "llm_rows": rows_llm,
        "threshold_rows": rows_th,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
