"""向量索引：FAISS + Embedding。

优先使用 DashScope text-embedding-v3（语义向量，质量高）。
若未配置 DASHSCOPE_API_KEY，则降级为 TF-IDF + TruncatedSVD 生成密集向量
（离线可用、零成本，但语义能力弱于真实 embedding，Recall 会偏低）。
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    CHUNKS_PATH, FAISS_INDEX_PATH, EMBED_MODEL, EMBED_DIM,
    EMBED_BATCH_SIZE, DASHSCOPE_API_KEY,
)


def _dashscope_embed(texts):
    """调用 DashScope embedding API。"""
    import dashscope
    from dashscope import TextEmbedding
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    dashscope.api_key = DASHSCOPE_API_KEY
    # ⚠️ text-embedding-v3 单次上限为 10 条（见 config.EMBED_BATCH_SIZE），
    #    超过会报 400 InvalidParameter: "batch size ... not larger than 10"
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = TextEmbedding.call(
            model=EMBED_MODEL,
            input=batch,
            text_type="document",
            dimension=EMBED_DIM,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding error: {resp.code} {resp.message}")
        # 保持与输入顺序一致
        embeddings = {e["text_index"]: e["embedding"] for e in resp.output["embeddings"]}
        for j in range(len(batch)):
            all_vecs.append(embeddings[j])
    return np.array(all_vecs, dtype=np.float32)


def _tfidf_svd_embed(texts, dim=EMBED_DIM, fitted=None):
    """离线兜底：TF-IDF + TruncatedSVD 降维到 dim 维。

    fitted: 若提供 (vectorizer, svd) 则直接 transform（查询时用）；
            否则 fit_transform（建索引时用）。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    if fitted is not None:
        vec, svd = fitted
        q = vec.transform(texts)
        X_svd = svd.transform(q)
        if X_svd.shape[1] < dim:
            pad = np.zeros((X_svd.shape[0], dim - X_svd.shape[1]), dtype=np.float32)
            X_svd = np.hstack([X_svd, pad])
        norms = np.linalg.norm(X_svd, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X_svd / norms).astype(np.float32)

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(texts)
    n_features = X.shape[1]
    actual_dim = min(dim, n_features - 1)
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    X_svd = svd.fit_transform(X)
    # 补齐到 EMBED_DIM（若 n_features 不足）
    if X_svd.shape[1] < dim:
        pad = np.zeros((X_svd.shape[0], dim - X_svd.shape[1]), dtype=np.float32)
        X_svd = np.hstack([X_svd, pad])
    # L2 归一化
    norms = np.linalg.norm(X_svd, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_svd = (X_svd / norms).astype(np.float32)
    return X_svd, (vec, svd)


def build_vector_index(chunks, use_dashscope=None, path=FAISS_INDEX_PATH):
    """构建 FAISS 向量索引。

    use_dashscope: None=自动判断（有 key 就用）, True/False=强制
    返回: (index, method)  —— method ∈ {"dashscope", "tfidf_svd"}
    查询时的 embedding 由 hybrid_retriever._embed_query 负责（同方法），
    本函数不返回 embed_fn。
    """
    import faiss

    texts = [c["text"] for c in chunks]
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)

    if use_dashscope:
        print(f"使用 DashScope {EMBED_MODEL} 生成 embedding ({len(texts)} 条)...")
        vectors = _dashscope_embed(texts)
        method = "dashscope"
    else:
        print("未配置 DASHSCOPE_API_KEY，使用 TF-IDF+SVD 离线兜底...")
        vectors, fitted = _tfidf_svd_embed(texts)
        # 缓存 fitted 模型，供查询时复用
        fitted_path = path.with_suffix(".tfidf_svd.pkl")
        import pickle
        with open(fitted_path, "wb") as f:
            pickle.dump(fitted, f)
        method = "tfidf_svd"

    dim = vectors.shape[1]
    print(f"向量维度: {dim}, 数量: {vectors.shape[0]}, 方法: {method}")

    index = faiss.IndexFlatIP(dim)  # 内积（已归一化 = 余弦相似度）
    index.add(vectors)
    # FAISS 在 Windows 不支持中文路径，用 serialize_index + 标准 open 写入
    with open(path, "wb") as f:
        f.write(faiss.serialize_index(index))
    print(f"FAISS 索引已写入 {path}")

    # 保存元数据：manifest（含 embedding 方法）+ chunks 列表
    #
    # ⚠️ 必须记录 method：DashScope(1024 维) 与 TF-IDF+SVD(补齐到 1024 维) 的
    #    维度**恰好相同**，若查询侧后端与建索引时不一致，FAISS 不会报错，
    #    只会返回无意义的相似度——这是最危险的一类静默错误。见 assert_backend_consistent()。
    meta_path = path.with_suffix(".meta.json")
    manifest = {
        "method": method,                      # "dashscope" | "tfidf_svd"
        "model": EMBED_MODEL if method == "dashscope" else "tfidf_svd",
        "dim": int(dim),
        "count": len(chunks),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunks": [{"chunk_id": c["chunk_id"], "kb_id": c["kb_id"],
                    "problem": c["problem"]} for c in chunks],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return index, method


def load_vector_index(path=FAISS_INDEX_PATH):
    import faiss
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    index = faiss.deserialize_index(data)
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def assert_backend_consistent(meta, use_dashscope=None):
    """校验「建索引时的 embedding 方法」与「查询侧当前后端」是否一致。

    为什么必须显式校验（而不是靠 FAISS 报维度错）：
        DashScope text-embedding-v3 → 1024 维
        TF-IDF + TruncatedSVD       → 补齐到 EMBED_DIM = 1024 维
    两者维度**恰好相同** → FAISS 的 IndexFlatIP.search 不会抛异常，
    只会拿不同语义空间的向量算内积，返回**无意义的相似度**。
    这类错误不报错、不崩溃，只是结果悄悄变差，因此必须主动拦截。

    meta: load_vector_index() 的第二个返回值（新版为 dict；旧版为 chunk list）
    不一致时抛 RuntimeError，并给出可执行的修复命令。
    """
    if use_dashscope is None:
        use_dashscope = bool(DASHSCOPE_API_KEY)
    expected = "dashscope" if use_dashscope else "tfidf_svd"

    # 旧格式（裸 list）→ 无 method 记录，无法校验，仅告警
    if not isinstance(meta, dict) or "method" not in meta:
        print("⚠️ faiss_index.meta.json 为旧格式（无 method 字段），无法校验 "
              "embedding 后端一致性。请重跑 src/vector_index.py 或运行 "
              "scripts/rebuild_index.py 生成新版 manifest。")
        return expected

    actual = meta.get("method")
    if actual != expected:
        raise RuntimeError(
            f"embedding 后端不一致：索引由 `{actual}` 构建"
            f"（model={meta.get('model')}, dim={meta.get('dim')}, "
            f"built_at={meta.get('built_at')}），但当前查询侧将使用 `{expected}`。\n"
            f"两者维度可能相同（{EMBED_DIM}），FAISS 不会报错，只会返回错误结果。\n"
            f"→ 修复：① 让查询侧改用 `{actual}`；或 ② 重跑 src/vector_index.py 重建索引。"
        )
    return expected


def load_fitted_embedder(path=None, strict=False):
    """加载缓存的 TF-IDF+SVD fitted 模型（离线兜底查询用）。

    path: 索引文件路径（*.bin）。fitted 模型与其**同名同后缀**落盘
          （`build_vector_index` 里 `path.with_suffix(".tfidf_svd.pkl")`），
          故默认取 `FAISS_INDEX_PATH`（cause 级索引）保持向后兼容。

    🔴 2026-10-05 修复的P0 既有 bug
    ------------------------------------------------------------------
    原实现**硬编码** `FAISS_INDEX_PATH`（cause 级索引 `faiss_index.bin`），
    但 kb 模式（B2，`RETRIEVER_MODE=kb`，当前默认）用的是
    `faiss_index_kb.bin`，其 fitted 模型落在 `faiss_index_kb.tfidf_svd.pkl`。
    于是 kb 模式下 `load_fitted_embedder()` 恒返回 `None`
    → 查询侧 `_tfidf_svd_embed([query], fitted=None)` 会**用单条 query
      重新 fit** 一个迷你 TF-IDF+SVD。实测后果有**两条**（对照实验见
      `exp/_tmp_bug_mechanism.py`）：

      (a) query 较短、TF-IDF 只切出 1 个 token → `n_components=0`
          → `InvalidParameterError: The 'n_components' parameter ... Got 0`
          这是**吵闹的**失败（会崩），原判断只覆盖了这一条；
      (b) query 较长、切出多个 token → fit 出 vocab=3 的迷你模型，
          其 1024 维向量查 16367 词表建的索引，**维度恰好相同所以 FAISS
          完全不报错**，静默返回无意义相似度（实测 top1 = KB002
          "Not Sticking to the Bed" cos=0.3613，而正确答案是
          KB006 "Stringing or Oozing"）。这是**安静的**失败，
          与 `assert_backend_consistent` 拦的属同一类，危害更大。

    为什么以前没暴露：生产有 `DASHSCOPE_API_KEY`，走 DashScope 分支
    **根本不读这个文件**。只有「无 key 的离线兜底」才会踩到。

    ⚠️ 另注：修好这个 bug 后会暴露一个**独立的既有缺陷**——离线兜底的
    TF-IDF 对**中文 query 完全失效**（索引语料是英文，且默认
    token_pattern 不做中文分词 → 命中 0 → 查询向量全零 → 双路皆零 →
    返回 KB001~KB005 的字母序）。中文场景必须配 `DASHSCOPE_API_KEY`，
    或给 TfidfVectorizer 换中文分析器后重建索引。

    strict=True 时，文件缺失**直接抛错**并给出修复命令，而不是返回 None
    让调用方在后续某个不相关的地方崩掉（SVD 越界报错完全指不到真因，
    且多词 query 根本不报错）。
    """
    import pickle
    base = Path(path) if path is not None else FAISS_INDEX_PATH
    fitted_path = base.with_suffix(".tfidf_svd.pkl")
    if not fitted_path.exists():
        if strict:
            raise FileNotFoundError(
                f"离线兜底向量检索缺少 fitted 模型：{fitted_path}\n"
                f"→ 修复：重跑 `python src/build_kb_index.py`"
                f"（kb 模式）或 `python src/vector_index.py`（cause 级模式）。\n"
                f"⚠️ 不要跳过这一步直接查询——那会用单条 query 重新 fit 出"
                f"迷你模型：短 query 抛「TruncatedSVD n_components 越界」"
                f"（报错指不到真因），长 query 则**静默返回无意义相似度**。"
            )
        return None
    with open(fitted_path, "rb") as f:
        return pickle.load(f)


def query_vector(index, query_embedding, top_k=5):
    """查询向量索引，返回 (scores, indices)。"""
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return scores[0], indices[0]


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    build_vector_index(chunks)


if __name__ == "__main__":
    main()
