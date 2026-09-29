"""实验：query 改写 + rerank 精排的四组对比（v2 策略）。

v1（全局应用）实测为负结果：改写 0.8000、精排 0.8125，均低于 baseline 0.8500。
v2 按 Advisor 建议改为「条件触发 + 双路分离」：
  - 改写：扩展术语只喂 BM25 路（bm25_query），向量路仍用原 query，保护 embedding 语义
  - 精排：置信度门控，仅当融合 top-1 余弦 < GATE_COSINE 时才触发 rerank

四组配置（同一评测集、同一检索底座）：
  A. baseline        —— 原双路融合（Recall@5 = 0.8500 基线）
  B. +rewrite(v2)    —— 双路分离改写
  C. +rerank(v2)     —— 门控精排（top-20 → gte-rerank-v2 → top-5）
  D. both            —— 改写 + 门控精排

指标：整体 Recall@5 + 分类别 Recall@5。
所有 LLM 调用带磁盘缓存，重跑不重复烧 token。
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EVAL_SET_PATH, RECALL_K
from hybrid_retriever import HybridRetriever
from query_rewriter import rewrite_query
from reranker import rerank, RERANK_CANDIDATES

RESULT_PATH = Path(__file__).resolve().parent / "exp_rewrite_rerank_result.json"

CONFIGS = {
    "A_baseline": {"rewrite": False, "rerank": False},
    "B_rewrite": {"rewrite": True, "rerank": False},
    "C_rerank": {"rewrite": False, "rerank": True},
    "D_both": {"rewrite": True, "rerank": True},
}

CATS = ["single_defect", "multi_defect", "param_compensation", "ambiguous"]


def run_one(retriever, eval_set, cfg):
    pos_hits = 0
    pos_total = 0
    cat_hits = defaultdict(int)
    cat_total = defaultdict(int)

    for item in eval_set:
        q = item["query"]
        gold = set(item["gold"])
        cat = item["category"]
        if not gold:
            continue  # 负例不参与 Recall（拒答是另一条评估线）

        # v2 改写：扩展术语只喂 BM25 路，向量路用原 query（双路分离）
        expansion = rewrite_query(q) if cfg["rewrite"] else ""
        bm25_q = f"{q} {expansion}" if expansion else q

        if cfg["rerank"]:
            # 召回放宽到 top-20，精排收紧到 top-5（margin 门控在 rerank 内部）。
            # 注意：若只对 top-5 精排，Recall@5 与 baseline 必然相同（集合命中不看顺序），
            # 必须从 top-20 召回才能把 6-20 位的 gold 救上来。
            candidates = retriever.retrieve(q, top_k=RERANK_CANDIDATES, bm25_query=bm25_q)
            retrieved = rerank(q, candidates, top_k=RECALL_K)
        else:
            retrieved = retriever.retrieve(q, top_k=RECALL_K, bm25_query=bm25_q)

        retrieved_kb = [r["kb_id"] for r in retrieved]
        pos_total += 1
        cat_total[cat] += 1
        if any(kb in gold for kb in retrieved_kb):
            pos_hits += 1
            cat_hits[cat] += 1

    recall = pos_hits / pos_total if pos_total else 0.0
    cat_recall = {c: cat_hits.get(c, 0) / cat_total.get(c, 1) for c in CATS}
    return {"recall_at_5": round(recall, 4), "pos_hits": pos_hits,
            "pos_total": pos_total, "cat_recall": {c: round(v, 4) for c, v in cat_recall.items()}}


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    retriever = HybridRetriever()

    all_results = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        print(f"\n=== {name} (rewrite={cfg['rewrite']}, rerank={cfg['rerank']}) ===")
        res = run_one(retriever, eval_set, cfg)
        res["elapsed_s"] = round(time.time() - t0, 1)
        all_results[name] = res
        print(f"  Recall@5 = {res['recall_at_5']}  ({res['pos_hits']}/{res['pos_total']})  {res['elapsed_s']}s")
        for c in CATS:
            print(f"    {c:22s}: {res['cat_recall'][c]}")

    # 汇总对比表
    print("\n" + "=" * 70)
    print(f"{'配置':14s} {'Recall@5':>9s}  " + "  ".join(f"{c[:12]:>12s}" for c in CATS))
    for name, res in all_results.items():
        cats = "  ".join(f"{res['cat_recall'][c]:>12.4f}" for c in CATS)
        print(f"{name:14s} {res['recall_at_5']:>9.4f}  {cats}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
