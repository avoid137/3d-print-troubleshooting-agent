"""TF-IDF + SVD 离线兜底路径的三路消融（gold 修正后重算）。

背景：eval_set 的 gold 修正（7 个 v1 悬空引用 → v2 编号）后，
所有基于旧 gold 的对照数字都必须重算，否则与其他材料口径不一致。

本脚本优先复用 data/_backup_tfidf_20260928/ 里的 fitted 模型与 FAISS 索引
（那是切 DashScope 之前的 TF-IDF 产物）；**若该备份不存在（如全新 clone），
则现场对同一批 chunk 重新 fit 一份 TF-IDF+SVD** —— 语料与 `random_state=42` 均相同，
指标结论不变。全程不调 API。

用法：
  python exp/exp_tfidf_ablation.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

CHUNKS = json.loads((PROJ / "data" / "chunks.json").read_text(encoding="utf-8"))
EVAL = json.loads((PROJ / "data" / "eval_set.json").read_text(encoding="utf-8"))
BAK = PROJ / "data" / "_backup_tfidf_20260928"
BAK_MODEL = BAK / "faiss_index.tfidf_svd.pkl"

from bm25_index import load_bm25, query_bm25          # noqa: E402
from vector_index import _tfidf_svd_embed             # noqa: E402

RRF_K = 60
TOP_K = 5

bm25 = load_bm25()
docs = [c["text"] for c in CHUNKS]
if BAK_MODEL.exists():
    fitted = pickle.load(open(BAK_MODEL, "rb"))
    D = _tfidf_svd_embed(docs, fitted=fitted)
    print(" [向量来源] 复用备份 fitted 模型")
else:
    D, fitted = _tfidf_svd_embed(docs)
    print(" [向量来源] 备份不存在 → 现场 fit（语料/seed 相同，结论不变）")
Q = _tfidf_svd_embed([e["query"] for e in EVAL], fitted=fitted)
print(f"TF-IDF+SVD 向量：doc {D.shape} / query {Q.shape}")


def ranks_bm25(q, n=50):
    return {idx: r for r, (idx, _) in enumerate(query_bm25(bm25, q, top_k=n))}


def eval_mode(mode, bw=0.5, vw=0.5):
    hits = total = 0
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
            order = [int(j) for j in np.argsort(-(Q[i] @ D.T))[:TOP_K]]
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
    cats = {c: round(cat_hits.get(c, 0) / cat_total.get(c, 1), 4) for c in cat_total}
    return hits, total, cats


print("=" * 64)
print("TF-IDF+SVD 离线兜底 · 三路消融（新 gold）")
print("=" * 64)
for mode, label in [("bm25", "bm25-only          "),
                    ("vector", "vector-only(tfidf) "),
                    ("hybrid", "hybrid 0.5/0.5     ")]:
    h, t, cats = eval_mode(mode)
    print(f"  {label}: {h}/{t} = {h / t:.4f}")
print()
print("说明：bm25-only 与 embedding 后端无关，应与 DashScope 方案下的 125/160 一致。")
