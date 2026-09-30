"""kb 级 vs chunk 级 索引 A/B 对比（仅检索层 Recall@5，秒级，不烧 LLM）。

A 组：chunk 级（现状，复用已建 faiss+bm25 索引，371 块）
B1 组：kb 级，每 kb 块仅聚合 cause 短语（短，向量聚焦）
B2 组：kb 级，每 kb 块聚合 cause+solution 全量（信息全，块长）

统一口径：同 HybridRetriever 的 RRF(0.6:0.4, K=60) + top-5；
         复用 src/evaluate.py 的 evaluate（按 top-5 返回单元的 kb_id 是否覆盖 gold 判命中）。
先跑 A 组校验是否复现 136/160=0.8500 基线，再信 B 组。
"""
import sys, json
from pathlib import Path

SRC = Path("D:/3D打印故障诊断Agent/src")
sys.path.insert(0, str(SRC))

import faiss
from config import KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL
from bm25_index import tokenize
from rank_bm25 import BM25Okapi
from hybrid_retriever import HybridRetriever
from vector_index import _dashscope_embed
from evaluate import evaluate


def build_kb_chunks(mode):
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
        else:  # b2：cause + solution
            body = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution', '')}"
                             for i, c in enumerate(causes))
        text = head + ("\n" + body if body else "")
        chunks.append({"chunk_id": item["kb_id"], "kb_id": item["kb_id"],
                       "problem": item["problem"], "type": "kb", "text": text})
    return chunks


def build_kb_index(chunks):
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    vecs = _dashscope_embed([c["text"] for c in chunks])  # DashScope 1024 维
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return bm25, index


def make_kb_retriever(chunks, bm25, vec_index):
    r = HybridRetriever(vector_backend="faiss")  # 加载 A 组索引作占位
    r.chunks = chunks
    r.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    r.bm25 = bm25
    r.vector_index = vec_index
    r.vector_meta = {"method": "dashscope", "model": EMBED_MODEL, "dim": EMBED_DIM}
    r.embed_method = "dashscope"
    return r


def main():
    kb_b1 = build_kb_chunks("b1")
    kb_b2 = build_kb_chunks("b2")
    print(f"kb chunks 数: B1={len(kb_b1)} B2={len(kb_b2)}")

    bm25_b1, vec_b1 = build_kb_index(kb_b1)
    bm25_b2, vec_b2 = build_kb_index(kb_b2)

    eval_set = json.load(open(EVAL_SET_PATH, encoding="utf-8"))

    print("\n########## A 组：chunk 级（现状基线，应 ≈0.8500）##########")
    r_a = HybridRetriever(vector_backend="faiss")
    res_a = evaluate(r_a, eval_set)

    print("\n########## B1 组：kb 级（仅 cause 短语）##########")
    res_b1 = evaluate(make_kb_retriever(kb_b1, bm25_b1, vec_b1), eval_set)

    print("\n########## B2 组：kb 级（cause+solution 全量）##########")
    res_b2 = evaluate(make_kb_retriever(kb_b2, bm25_b2, vec_b2), eval_set)

    print("\n========== 汇总（Recall@5，160 正例）==========")
    for name, res in [("A chunk级(现状)", res_a),
                      ("B1 kb级(短语)", res_b1),
                      ("B2 kb级(全量)", res_b2)]:
        print(f"  {name:18s}: {res['recall_at_k']:.4f}  "
              f"({res['pos_hits']}/{res['pos_total']})")


if __name__ == "__main__":
    main()
