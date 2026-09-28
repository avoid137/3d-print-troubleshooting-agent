"""P0 评测：计算 Recall@5 及各类别指标。

指标定义：
  Recall@5（正例）：top-5 中任一 chunk 的 kb_id ∈ gold → 命中
  对多缺陷（gold 有多个）：top-5 覆盖 gold 中任意一个即命中（宽松）
  对库外负例（gold=[]）：统计「最高置信度 chunk 的分数」，越低越好（应拒答）
"""
import json
import time
from collections import defaultdict
from config import EVAL_SET_PATH, RECALL_K
from hybrid_retriever import HybridRetriever


def evaluate(retriever, eval_set, top_k=RECALL_K):
    results = []
    pos_hits = 0
    pos_total = 0
    neg_count = 0
    neg_max_scores = []  # 负例的 top-1 分数（越低越像拒答）

    cat_hits = defaultdict(int)
    cat_total = defaultdict(int)

    t0 = time.time()
    for item in eval_set:
        q = item["query"]
        gold = set(item["gold"])
        cat = item["category"]

        retrieved = retriever.retrieve(q, top_k=top_k)
        retrieved_kb_ids = [r["kb_id"] for r in retrieved]
        top1_score = retrieved[0]["score"] if retrieved else 0.0

        if gold:
            # 正例
            pos_total += 1
            cat_total[cat] += 1
            hit = any(kb in gold for kb in retrieved_kb_ids)
            if hit:
                pos_hits += 1
                cat_hits[cat] += 1
            results.append({"id": item["id"], "category": cat, "hit": hit,
                            "gold": list(gold), "retrieved": retrieved_kb_ids})
        else:
            # 负例
            neg_count += 1
            neg_max_scores.append(top1_score)

    elapsed = time.time() - t0
    recall = pos_hits / pos_total if pos_total else 0.0

    print("=" * 60)
    print(f"P0 评测结果（共 {len(eval_set)} 条，耗时 {elapsed:.1f}s）")
    print("=" * 60)
    print(f"正例 Recall@{top_k}: {pos_hits}/{pos_total} = {recall:.4f}")
    print(f"负例数: {neg_count}")
    if neg_max_scores:
        import numpy as np
        arr = np.array(neg_max_scores)
        print(f"负例 top-1 分数: mean={arr.mean():.4f} max={arr.max():.4f} min={arr.min():.4f}")
    print()
    print("各类别 Recall@" + str(top_k) + ":")
    for cat in ["single_defect", "multi_defect", "param_compensation", "ambiguous"]:
        t = cat_total.get(cat, 0)
        h = cat_hits.get(cat, 0)
        r = h / t if t else 0.0
        print(f"  {cat:25s}: {h}/{t} = {r:.4f}")
    print(f"  {'out_of_scope':25s}: 拒答评估（见上 top-1 分数）")

    return {
        "recall_at_k": recall,
        "pos_hits": pos_hits,
        "pos_total": pos_total,
        "cat_recall": {c: cat_hits.get(c, 0) / cat_total.get(c, 1)
                       for c in cat_total},
    }


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    retriever = HybridRetriever()
    evaluate(retriever, eval_set)


if __name__ == "__main__":
    main()
