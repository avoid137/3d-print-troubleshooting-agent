"""双路加权融合检索：BM25 + 向量，用 RRF（Reciprocal Rank Fusion）合并。

RRF 对分数尺度不敏感（BM25 分数和余弦相似度不在同一量纲），
是 hybrid search 的工业标准做法。
"""
import json
import numpy as np
from config import (
    CHUNKS_PATH, BM25_WEIGHT, VECTOR_WEIGHT, TOP_K, EMBED_DIM,
    DASHSCOPE_API_KEY, EMBED_MODEL,
)
from bm25_index import load_bm25, query_bm25
from vector_index import load_vector_index, load_fitted_embedder, _tfidf_svd_embed

RRF_K = 60  # RRF 常数，业界常用 60

# 进程内缓存 fitted embedder（离线兜底）
_FITTED_EMBEDDER = None


def _embed_query(query, use_dashscope=None):
    """对 query 做 embedding（与建索引时同方法）。"""
    global _FITTED_EMBEDDER
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)
    if use_dashscope:
        import dashscope
        from dashscope import TextEmbedding
        dashscope.api_key = DASHSCOPE_API_KEY
        resp = TextEmbedding.call(
            model=EMBED_MODEL, input=query, text_type="query",
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding error: {resp.code} {resp.message}")
        vec = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
    else:
        # 复用建索引时缓存的 TF-IDF+SVD fitted 模型
        if _FITTED_EMBEDDER is None:
            _FITTED_EMBEDDER = load_fitted_embedder()
        vec = _tfidf_svd_embed([query], fitted=_FITTED_EMBEDDER)[0]
    return vec


class HybridRetriever:
    def __init__(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.bm25 = load_bm25()
        self.vector_index, self.vector_meta = load_vector_index()

    def retrieve(self, query, top_k=TOP_K, bm25_weight=BM25_WEIGHT,
                 vector_weight=VECTOR_WEIGHT):
        # 1. BM25 路：取 top-50 候选
        bm25_top = query_bm25(self.bm25, query, top_k=50)
        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_top)}

        # 2. 向量路：取 top-50 候选
        qvec = _embed_query(query)
        scores, indices = self.vector_index.search(
            qvec.reshape(1, -1), 50)
        vec_rank = {int(idx): r for r, idx in enumerate(indices[0])}

        # 3. RRF 加权融合
        all_idx = set(bm25_rank) | set(vec_rank)
        fused = {}
        for idx in all_idx:
            s = 0.0
            if idx in bm25_rank:
                s += bm25_weight / (RRF_K + bm25_rank[idx])
            if idx in vec_rank:
                s += vector_weight / (RRF_K + vec_rank[idx])
            fused[idx] = s

        # 4. 排序取 top_k
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "kb_id": c["kb_id"],
                "problem": c["problem"],
                "type": c["type"],
                "score": round(score, 6),
                "text": c["text"][:200],
            })
        return results

    def retrieve_bm25_only(self, query, top_k=TOP_K):
        bm25_top = query_bm25(self.bm25, query, top_k=top_k)
        return [self.chunks[i] for i, _ in bm25_top]

    def retrieve_vector_only(self, query, top_k=TOP_K):
        qvec = _embed_query(query)
        _, indices = self.vector_index.search(qvec.reshape(1, -1), top_k)
        return [self.chunks[int(i)] for i in indices[0]]


def main():
    r = HybridRetriever()
    queries = [
        "my print has stringing and oozing",
        "the layers are shifting",
        "warping corners lifting off bed",
        "nozzle clogged filament not coming out",
    ]
    for q in queries:
        print(f"\n=== Query: {q} ===")
        for res in r.retrieve(q):
            print(f"  [{res['kb_id']}] {res['problem']} ({res['type']}) score={res['score']}")
            print(f"      {res['text'][:80]}...")


if __name__ == "__main__":
    main()
