"""零 LLM 开销：top-k=10 vs top-k=5 的 gold 检索覆盖对比（B2 kb 级索引）。

目的：量化选项②（top-k=10）这个杠杆——那 29 道「gold 没进 top-5」的题，
有多少能在 k=10 进上下文。决定要不要烧 ~90min LLM 跑生成实验。
只做 BM25 + FAISS 检索（无 API 调用），秒级。
"""
import json
import sys
from collections import Counter

sys.path.insert(0, "src")
import faiss  # noqa: E402
from config import KB_V2_PATH, EMBED_DIM, EMBED_MODEL  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402

kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
chunks = []
for it in kb:
    causes = it.get("causes") or []
    body = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution', '')}"
                     for i, c in enumerate(causes))
    chunks.append({"chunk_id": it["kb_id"], "kb_id": it["kb_id"],
                   "problem": it["problem"], "type": "kb",
                   "text": it["problem"] + ("\n" + body if body else "")})
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
vecs = _dashscope_embed([c["text"] for c in chunks])
idx = faiss.IndexFlatIP(vecs.shape[1])
idx.add(vecs)
r = HybridRetriever(vector_backend="faiss")
r.chunks = chunks
r.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
r.bm25 = bm25
r.vector_index = idx
r.vector_meta = {"method": "dashscope", "model": EMBED_MODEL, "dim": EMBED_DIM}
r.embed_method = "dashscope"

ev = json.load(open("data/eval_set.json", encoding="utf-8"))
in_scope = [e for e in ev if e["gold"]]
cov5 = cov10 = 0
recovered = []
still_gap = []
for e in in_scope:
    g = set(e["gold"])
    top5 = [x["chunk_id"] for x in r.retrieve(e["query"], top_k=5)]
    top10 = [x["chunk_id"] for x in r.retrieve(e["query"], top_k=10)]
    if g.issubset(set(top5)):
        cov5 += 1
    if g.issubset(set(top10)):
        cov10 += 1
    else:
        still_gap.append(e["id"])
    if (not g.issubset(set(top5))) and g.issubset(set(top10)):
        recovered.append(e["id"])
n = len(in_scope)
print(f"in_scope = {n}")
print(f"gold⊆top5  覆盖 = {cov5}/{n} = {cov5/n:.4f}")
print(f"gold⊆top10 覆盖 = {cov10}/{n} = {cov10/n:.4f}")
print(f"k=5 缺口 = {n-cov5} 题；其中 k=10 救回 = {len(recovered)} 题")
print(f"  k=10 救回 ID: {recovered}")
print(f"k=10 仍缺口 = {n-cov10} 题: {still_gap}")
cat = Counter(e["category"] for e in ev if e["id"] in recovered)
print(f"k=10 救回按类别: {dict(cat)}")
