"""双路加权融合检索：BM25 + 向量，用 RRF（Reciprocal Rank Fusion）合并。

RRF 对分数尺度不敏感（BM25 分数和余弦相似度不在同一量纲），
是 hybrid search 的工业标准做法。

向量后端可切换（vector_backend）：
  - "faiss"（默认）：IndexFlatIP 暴力精确检索，无元数据过滤能力。
  - "chroma"：PersistentClient 持久化索引，支持 where 元数据过滤
    （如 {"type": "cause"} / {"kb_id": "KB006"}）。
  两者共用同一套 embedding，已实测等价（exp/exp_chroma_eval.py，identical=true）。
"""
import json
import numpy as np
from config import (
    CHUNKS_PATH, BM25_WEIGHT, VECTOR_WEIGHT, TOP_K, EMBED_DIM,
    DASHSCOPE_API_KEY, EMBED_MODEL, VECTOR_BACKEND,
    KB_CHUNKS_PATH, KB_BM25_INDEX_PATH, KB_FAISS_INDEX_PATH, RETRIEVER_MODE,
)
from bm25_index import load_bm25, query_bm25
from vector_index import (
    load_vector_index, load_fitted_embedder, _tfidf_svd_embed,
    assert_backend_consistent,
)

RRF_K = 60  # RRF 常数，业界常用 60

# 进程内缓存 fitted embedder（离线兜底）。
# 🔴 2026-10-05：由「单个全局变量」改为**按索引路径分桶**。
# 原实现只有一个 `_FITTED_EMBEDDER`，一旦某个mode 先填了缓存，
# 另一个 mode 会复用**错误的** fitted 模型（两个索引的 vocab/SVD 基不同），
# 静默返回无意义的相似度——与 assert_backend_consistent 拦的是同一类错误。
_FITTED_EMBEDDER_CACHE = {}


def _reset_fitted_embedder_cache():
    """清空 fitted embedder 缓存（测试用；生产无需调用）。"""
    _FITTED_EMBEDDER_CACHE.clear()


def _embed_query(query, use_dashscope=None, fitted_path=None):
    """对 query 做embedding（与建索引时同方法）。

    fitted_path: **必须**传当前实例实际加载的索引路径（*.bin）。
        🔴 2026-10-05 修复：原先固定读 `FAISS_INDEX_PATH`（cause 级索引），
        而 kb 模式下 fitted 模型在 `faiss_index_kb.tfidf_svd.pkl` → 恒拿到 None。
        取不到时的后果**不只是报错**，实测有两条（见 `exp/_tmp_bug_mechanism.py`）：
          (a) 单词 query：TF-IDF 只切出 1 个 token → n_components=0
              → `InvalidParameterError` 抛错（吵闹，原判断对）；
          (b) 多词 query：fit 出vocab=3 的迷你模型，其向量查 16367 词表建的索引
              → 维度同为 1024，FAISS **不报错**，静默返回无意义相似度
              （如 0.3355，与 assert_backend_consistent 拦的同类，且更隐蔽）。
        现在按路径取，且**取不到就报错**，不再静默重新 fit。
    """
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)
    if use_dashscope:
        import dashscope
        from dashscope import TextEmbedding
        dashscope.api_key = DASHSCOPE_API_KEY
        resp = TextEmbedding.call(
            model=EMBED_MODEL, input=query, text_type="query",
            dimension=EMBED_DIM,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding error: {resp.code} {resp.message}")
        vec = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
    else:
        # 复用建索引时缓存的 TF-IDF+SVD fitted 模型（按索引路径分桶）
        if fitted_path is None:
            raise ValueError(
                "离线兜底（无 DASHSCOPE_API_KEY）时必须传 fitted_path，"
                "否则无法确定用哪套索引的 TF-IDF+SVD 模型。"
            )
        key = str(fitted_path)
        if key not in _FITTED_EMBEDDER_CACHE:
            # strict=True：文件缺失直接抛 FileNotFoundError 并给修复命令，
            # 而不是返回 None 让 _tfidf_svd_embed 用单条 query 重新 fit
            # ——后者对单词 query 会抛「TruncatedSVD n_components 越界」，
            # 但对多词 query **不抛错**，只静默返回跨语义空间的假相似度，
            # 报错完全指不到真因（单词时也是错的报错）。
            _FITTED_EMBEDDER_CACHE[key] = load_fitted_embedder(
                fitted_path, strict=True)
        vec = _tfidf_svd_embed([query],
                               fitted=_FITTED_EMBEDDER_CACHE[key])[0]
    return vec


def _match_where(chunk, where):
    """BM25 路的元数据过滤：仅支持扁平相等匹配（如 {"type": "cause"}）。

    与 Chroma where 的简单形态对齐；遇到 $and/$or 等算子直接报错，
    避免静默忽略过滤条件导致双路结果不一致。
    """
    if not where:
        return True
    for k, v in where.items():
        if k.startswith("$"):
            raise ValueError(
                f"BM25 路仅支持扁平相等 where，不支持算子 {k}；"
                f"请简化过滤条件或改用向量单路（chroma）")
        if chunk.get(k) != v:
            return False
    return True


class HybridRetriever:
    def __init__(self, vector_backend=VECTOR_BACKEND, mode=None):
        if vector_backend not in ("faiss", "chroma"):
            raise ValueError(f"vector_backend 只能是 faiss / chroma，收到 {vector_backend}")
        self.vector_backend = vector_backend
        # mode: "kb"(默认, B2 级每条 kb 1 块) / "chunk"(旧 cause 级)；决定加载哪套索引
        self.mode = mode or RETRIEVER_MODE
        if self.mode == "kb":
            chunks_path = KB_CHUNKS_PATH
            bm25_path = KB_BM25_INDEX_PATH
            faiss_path = KB_FAISS_INDEX_PATH
        else:
            chunks_path = CHUNKS_PATH
            bm25_path = BM25_INDEX_PATH
            faiss_path = FAISS_INDEX_PATH
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.bm25 = load_bm25(bm25_path)
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self._chroma_collection = None
        # 🔴 2026-10-05：记住本实例实际用的索引路径，查询时按它取fitted 模型。
        # 之前 `_embed_query` 硬编码读cause 级索引 → kb 模式下取不到。
        self.faiss_path = faiss_path
        if vector_backend == "faiss":
            self.vector_index, self.vector_meta = load_vector_index(faiss_path)
            # 启动即校验：索引的 embedding 方法与查询侧后端必须一致，
            # 否则维度相同会静默返回错误相似度（详见 vector_index.assert_backend_consistent）
            self.embed_method = assert_backend_consistent(self.vector_meta)
        else:
            # chroma 后端：懒加载 collection（首次查询时连接）
            self.vector_index, self.vector_meta = None, None
            self.embed_method = "chroma"

    def _get_chroma_collection(self):
        if self._chroma_collection is None:
            from chroma_index import load_chroma
            _, self._chroma_collection = load_chroma()
        return self._chroma_collection

    def _vector_search(self, query, top_n, where=None):
        """向量路统一出口：返回 [(chunk 位置下标, 余弦相似度), ...]。"""
        if self.vector_backend == "faiss":
            if where:
                raise ValueError("faiss 后端不支持 where 过滤，请用 vector_backend='chroma'")
            qvec = _embed_query(query, fitted_path=self.faiss_path)
            scores, indices = self.vector_index.search(qvec.reshape(1, -1), top_n)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0])]
        # chroma 后端
        collection = self._get_chroma_collection()
        qvec = _embed_query(query, fitted_path=self.faiss_path)
        kwargs = {"query_embeddings": [qvec.tolist()], "n_results": top_n,
                  "include": ["distances"]}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            idx = self.chunk_id_to_idx.get(cid)
            if idx is None:  # chroma 与 chunks.json 不同步时的防御
                continue
            out.append((idx, 1.0 - float(dist)))  # cosine distance → similarity
        return out

    def retrieve(self, query, top_k=TOP_K, bm25_weight=BM25_WEIGHT,
                 vector_weight=VECTOR_WEIGHT, where=None, bm25_query=None):
        # bm25_query：BM25 路的独立查询串（默认同 query）。
        # 用途：query 改写时，扩展术语只喂 BM25 路（词面扩展），
        #       向量路仍用原 query（保护 embedding 语义不被稀释）。
        if bm25_query is None:
            bm25_query = query

        # 1. BM25 路：取 top-50 候选（同时保留原始分，供拒答判断用）
        #    where 非空时双路同步过滤，保证融合结果口径一致
        bm25_top = query_bm25(self.bm25, bm25_query, top_k=50)
        if where:
            bm25_top = [(i, s) for i, s in bm25_top
                        if _match_where(self.chunks[i], where)]
        bm25_rank, bm25_raw = {}, {}
        for r, (idx, sc) in enumerate(bm25_top):
            bm25_rank[idx] = r
            bm25_raw[idx] = float(sc)

        # 2. 向量路：取 top-50 候选（同时保留余弦相似度）
        vec_rank, vec_raw = {}, {}
        for r, (idx, cos) in enumerate(self._vector_search(query, 50, where=where)):
            vec_rank[idx] = r
            vec_raw[idx] = cos

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
        #
        # ⚠️ score 是 RRF 融合分，**只反映排名、不反映相似度**（top-1 恒为常数），
        #    因此**不能用来做拒答阈值**。拒答请用 bm25_score / cosine 原始分。
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
                "bm25_score": round(bm25_raw.get(idx, 0.0), 4),
                "cosine": round(vec_raw.get(idx, -1.0), 4),
                "text": c["text"][:200],
            })
        return results

    def retrieve_bm25_only(self, query, top_k=TOP_K):
        bm25_top = query_bm25(self.bm25, query, top_k=top_k)
        return [self.chunks[i] for i, _ in bm25_top]

    def retrieve_vector_only(self, query, top_k=TOP_K, where=None):
        return [self.chunks[idx] for idx, _ in
                self._vector_search(query, top_k, where=where)]


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
