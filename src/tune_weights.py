"""权重调优：寻找 BM25 vs Vector 的最优 RRF 权重组合。"""
import json
from config import EVAL_SET_PATH
from hybrid_retriever import HybridRetriever


def recall_with_weights(retriever, eval_set, bm25_w, vec_w, top_k=5):
    hits = 0
    total = 0
    for item in eval_set:
        gold = set(item["gold"])
        if not gold:
            continue
        total += 1
        retrieved = retriever.retrieve(item["query"], top_k, bm25_w, vec_w)
        kb_ids = [r["kb_id"] for r in retrieved]
        if any(kb in gold for kb in kb_ids):
            hits += 1
    return hits / total if total else 0.0


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    r = HybridRetriever()

    print("权重调优 (BM25 : Vector) -> Recall@5")
    print("-" * 40)
    best = (0, 0, 0)
    for bw in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        vw = 1.0 - bw
        recall = recall_with_weights(r, eval_set, bw, vw)
        marker = " <-- best" if recall > best[2] else ""
        print(f"  {bw:.1f} : {vw:.1f}  ->  {recall:.4f}{marker}")
        if recall > best[2]:
            best = (bw, vw, recall)
    print(f"\n最优: BM25={best[0]} Vector={best[1]} Recall@5={best[2]:.4f}")


if __name__ == "__main__":
    main()
