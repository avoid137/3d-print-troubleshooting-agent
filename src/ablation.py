"""消融实验：BM25-only / Vector-only / Hybrid 三路对比。"""
import json
from config import EVAL_SET_PATH, RECALL_K
from hybrid_retriever import HybridRetriever


def eval_only(retriever, eval_set, mode, top_k=RECALL_K):
    hits = 0
    total = 0
    for item in eval_set:
        gold = set(item["gold"])
        if not gold:
            continue
        total += 1
        if mode == "bm25":
            retrieved = retriever.retrieve_bm25_only(item["query"], top_k)
        elif mode == "vector":
            retrieved = retriever.retrieve_vector_only(item["query"], top_k)
        else:
            retrieved = retriever.retrieve(item["query"], top_k)
        kb_ids = [r["kb_id"] if isinstance(r, dict) else r["kb_id"]
                  for r in retrieved]
        if any(kb in gold for kb in kb_ids):
            hits += 1
    return hits / total if total else 0.0, hits, total


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    r = HybridRetriever()

    print("消融实验：Recall@" + str(RECALL_K) + " 对比")
    print("-" * 50)
    for mode in ["bm25", "vector", "hybrid"]:
        recall, hits, total = eval_only(r, eval_set, mode)
        print(f"  {mode:10s}: {hits}/{total} = {recall:.4f}")


if __name__ == "__main__":
    main()
