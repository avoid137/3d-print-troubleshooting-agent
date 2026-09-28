"""补充实验：Hybrid（BM25 + DashScope 向量，RRF 融合）vs 现状。

复用 exp/ 下已缓存的 DashScope 向量，避免重复调用 API。
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"D:\wb-workspace\2026-09-07-11-07-13\3D打印故障诊断Agent")
sys.path.insert(0, str(PROJ / "src"))

CHUNKS = json.loads((PROJ / "data" / "chunks.json").read_text(encoding="utf-8"))
EVAL = json.loads((PROJ / "data" / "eval_set.json").read_text(encoding="utf-8"))
D = np.load(PROJ / "exp" / "doc_emb_dashscope.npy")
Q = np.load(PROJ / "exp" / "query_emb_dashscope.npy")

from bm25_index import load_bm25, query_bm25
bm25 = load_bm25()

RRF_K = 60
TOP_K = 5


def ranks_bm25(q, n=50):
    res = query_bm25(bm25, q, top_k=n)
    return {idx: r for r, (idx, _) in enumerate(res)}


def eval_mode(mode, bw, vw):
    hits = 0
    total = 0
    cat_hits, cat_total = {}, {}
    for i, item in enumerate(EVAL):
        gold = set(item["gold"])
        if not gold:
            continue
        total += 1
        cat = item["category"]

        if mode == "bm25":
            order = list(ranks_bm25(item["query"]).keys())[:TOP_K]
        elif mode == "vector":
            order = np.argsort(-(Q[i] @ D.T))[:TOP_K]
        else:
            rb = ranks_bm25(item["query"])
            rv = {int(j): r for r, j in enumerate(np.argsort(-(Q[i] @ D.T))[:50])}
            fused = {}
            for idx in set(rb) | set(rv):
                s = 0.0
                if idx in rb:
                    s += bw / (RRF_K + rb[idx])
                if idx in rv:
                    s += vw / (RRF_K + rv[idx])
                fused[idx] = s
            order = [x for x, _ in sorted(fused.items(), key=lambda t: -t[1])[:TOP_K]]

        got = [CHUNKS[j]["kb_id"] for j in order]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if any(k in gold for k in got):
            hits += 1
            cat_hits[cat] = cat_hits.get(cat, 0) + 1
    return hits / total, hits, total, {c: cat_hits.get(c, 0) / cat_total.get(c, 1) for c in cat_total}


print("=" * 64)
print("BM25 + DashScope 向量 · 消融与权重扫描")
print("=" * 64)
r, h, t, cats = eval_mode("bm25", 1, 0)
print(f"  bm25-only          : {h}/{t} = {r:.4f}")
r, h, t, cats = eval_mode("vector", 0, 1)
print(f"  vector-only(ds)    : {h}/{t} = {r:.4f}")
print()
print("  -- Hybrid 权重扫描 --")
best = None
for bw in [0.3, 0.4, 0.5, 0.6, 0.7]:
    r, h, t, cats = eval_mode("hybrid", bw, round(1 - bw, 1))
    flag = ""
    if best is None or r > best[0]:
        best = (r, bw, cats)
        flag = " <-- best"
    print(f"  {bw:.1f} : {round(1-bw,1):.1f}  ->  {r:.4f}{flag}")

print()
print(f"最优 Hybrid: BM25={best[1]:.1f} Vector={round(1-best[1],1):.1f} Recall@5={best[0]:.4f}")
print("  各类别:", {k: round(v, 4) for k, v in best[2].items()})
print()
print("--- 对照：TF-IDF+SVD 兜底方案（重算见 exp_tfidf_ablation.py）---")
print("  bm25-only          : 125/160 = 0.7812")
print("  vector-only(tfidf) : 117/160 = 0.7312")
print("  hybrid 0.5/0.5     : 125/160 = 0.7812   ← 与 BM25 单路持平，同质融合零增量")
