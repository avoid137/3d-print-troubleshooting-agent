"""ChromaDB 持久化索引：元数据过滤服务层。

**为什么需要它（与 FAISS 分工）**：
  FAISS 的 IndexFlatIP 只管「向量 → 最近邻」，**不存元数据、不支持 where 过滤**。
  而本项目过滤是刚需——「只看 FDM 的」「只看 Stringing 相关的」「只看 cause 级」。
  Chroma 用 PersistentClient 一行落盘，把 chunk 的元数据（kb_id / type / problem）塞进 metadata，
  就能在向量检索之上叠加 where 过滤。

**embedding 一致性**：
  与 FAISS 索引共用同一套 embedding（DashScope text-embedding-v3，兜底 TF-IDF+SVD）。
  兜底路径下 build 时保存 fitted（vectorizer+SVD），query 时复用同一模型——
  否则 query 会重新 fit、语义空间错乱，这是最容易踩的坑。

**落盘**：data/chroma_db/（D 盘），.gitignore 已排除（可重建产物）。
"""
import pickle

import numpy as np

from config import (
    CHUNKS_PATH, CHROMA_DB_PATH, CHROMA_COLLECTION, TOP_K, DASHSCOPE_API_KEY,
)
from vector_index import _dashscope_embed, _tfidf_svd_embed

# 兜底路径下 fitted 模型的缓存位置（build 时写入、query 时读取）
_FITTED_PATH = CHROMA_DB_PATH / "tfidf_fitted.pkl"


def _embed_texts(texts, use_dashscope=None):
    """对一批文本生成 embedding（与 FAISS 建索引同方法），返回 (N, dim)。

    TF-IDF 兜底时把 fitted 模型落盘，供 query 复用。
    """
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)
    if use_dashscope:
        return _dashscope_embed(texts)
    vectors, fitted = _tfidf_svd_embed(texts)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    with open(_FITTED_PATH, "wb") as f:
        pickle.dump(fitted, f)
    return vectors


def _embed_query_one(text, use_dashscope=None):
    """对单条 query 生成 embedding，返回一维 (dim,)。"""
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)
    if use_dashscope:
        return _dashscope_embed([text])[0].reshape(-1)
    if not _FITTED_PATH.exists():
        raise RuntimeError(
            f"TF-IDF fitted 模型缺失：{_FITTED_PATH}。请先运行 build_chroma() 建立索引。"
        )
    with open(_FITTED_PATH, "rb") as f:
        fitted = pickle.load(f)
    vec = _tfidf_svd_embed([text], fitted=fitted)  # shape (1, dim)
    return vec[0].reshape(-1)


def _metadata_of(chunk):
    """chunk → Chroma metadata（值必须为 str/int/float/bool，不能为 None）。"""
    meta = {
        "kb_id": chunk["kb_id"],
        "problem": chunk["problem"],
        "type": chunk["type"],
        "chunk_id": chunk["chunk_id"],
    }
    if chunk.get("cause"):
        meta["cause"] = chunk["cause"]
    return meta


def build_chroma(chunks=None, use_dashscope=None, reset=True):
    """建立（或重建）Chroma 持久化索引。

    chunks: None=读 chunks.json；否则传入 chunk 列表
    reset: True=先删除同名 collection 再重建（幂等）；False=get_or_create
    """
    import chromadb
    import json

    if chunks is None:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"生成 embedding（{len(texts)} 条，方法={'dashscope' if use_dashscope else 'auto'}）...")
    vectors = _embed_texts(texts, use_dashscope=use_dashscope)
    print(f"向量维度: {vectors.shape[1]}, 数量: {vectors.shape[0]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # 向量已 L2 归一化 → cosine 空间
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[_metadata_of(c) for c in chunks],
    )
    print(f"Chroma collection `{CHROMA_COLLECTION}` 已写入 {CHROMA_DB_PATH}，共 {len(chunks)} 条")
    return client, collection


def load_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(CHROMA_COLLECTION)
    return client, collection


def query_chroma(query_text, where=None, where_document=None, top_k=TOP_K,
                 use_dashscope=None):
    """带元数据过滤的向量查询。

    where:          元数据过滤，如 {"type": "cause"} 或 {"kb_id": "KB006"}
    where_document: 全文过滤，如 {"$contains": "stringing"}
    返回 list[dict]：{chunk_id, kb_id, problem, type, similarity, text}
    """
    client, collection = load_chroma()
    qvec = _embed_query_one(query_text, use_dashscope=use_dashscope)

    kwargs = {"query_embeddings": [qvec.tolist()], "n_results": top_k}
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document

    res = collection.query(**kwargs)
    ids = res["ids"][0]
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

    out = []
    for i, cid in enumerate(ids):
        m = metadatas[i] or {}
        out.append({
            "chunk_id": cid,
            "kb_id": m.get("kb_id", ""),
            "problem": m.get("problem", ""),
            "type": m.get("type", ""),
            # cosine space 下 distance 越小越相似（近似），转成相似度便于阅读
            "similarity": round(1.0 - (distances[i] or 0.0), 4),
            "text": (documents[i] or "")[:120],
        })
    return out


def main():
    client, collection = build_chroma()

    print("\n=== 无过滤 top-5：'nozzle clogged' ===")
    for r in query_chroma("nozzle clogged filament not coming out"):
        print(f"  [{r['kb_id']}] {r['problem']} ({r['type']}) sim={r['similarity']}")

    print("\n=== 过滤 type=cause：'nozzle clogged' ===")
    for r in query_chroma("nozzle clogged filament not coming out", where={"type": "cause"}):
        print(f"  [{r['kb_id']}] {r['problem']} ({r['type']}) sim={r['similarity']}")

    print("\n=== 过滤 kb_id=KB006：'printer has thin strings' ===")
    for r in query_chroma("printer has thin strings on the part", where={"kb_id": "KB006"}):
        print(f"  [{r['kb_id']}] {r['problem']} ({r['type']}) sim={r['similarity']}")


if __name__ == "__main__":
    main()