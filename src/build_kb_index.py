"""构建 B2 kb 级索引并落盘（data/kb_chunks.json + faiss_index_kb.bin + bm25_index_kb.pkl）。

B2 格式：每条 kb = 1 块（56 块），text = problem + intro + 全部「根因: 方案」。
这是 9/30 端到端验证的最优组合索引（检索 Recall@5 85%→90%，strict 43.75%→49.38%）。

用法：
    python src/build_kb_index.py
环境变量：
    RETRIEVER_MODE=chunk 时本脚本仍构建 kb 索引（与运行模式无关，索引是独立产物）。
"""
import json
import faiss  # noqa: F401（build_vector_index 内部用）

from config import (
    KB_V2_PATH, KB_CHUNKS_PATH, KB_FAISS_INDEX_PATH, KB_BM25_INDEX_PATH,
    EMBED_DIM, EMBED_MODEL,
)
from bm25_index import build_bm25
from vector_index import build_vector_index


def build_kb_chunks(mode="b2"):
    """每条 kb → 1 个 chunk。mode='b2' 含 cause+solution 全量；'b1' 仅 cause 短语。"""
    kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
    chunks = []
    for item in kb:
        causes = item.get("causes", []) or []
        intro = (item.get("intro") or "").strip()
        head = item["problem"]
        if intro:
            head += "\n" + intro
        if mode == "b1":
            body = "\n".join(f"{i+1}. {c['cause']}" for i, c in enumerate(causes))
        else:  # b2：cause + solution 全量
            body = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution', '')}"
                             for i, c in enumerate(causes))
        text = head + ("\n" + body if body else "")
        chunks.append({"chunk_id": item["kb_id"], "kb_id": item["kb_id"],
                       "problem": item["problem"], "type": "kb", "text": text})
    return chunks


def main():
    chunks = build_kb_chunks("b2")
    KB_CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"kb_chunks.json 写入 {KB_CHUNKS_PATH} ({len(chunks)} 块)")
    build_bm25(chunks, path=KB_BM25_INDEX_PATH)
    build_vector_index(chunks, path=KB_FAISS_INDEX_PATH)
    print("B2 kb 级索引构建完成（faiss + bm25 已落盘）")


if __name__ == "__main__":
    main()
