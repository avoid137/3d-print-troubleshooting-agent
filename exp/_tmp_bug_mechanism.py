# -*- coding: utf-8 -*-
"""P0 bug 机制归因对照实验（一次性，归因定稿用）。

结论：**原判断「单条 query 重新 fit → TruncatedSVD n_components 越界抛」只对了一半，
且漏掉了危害更严重的一层。** 本脚本用四组对照把真实机制钉死。

用法（在仓库根）：
    .venv/Scripts/python.exe exp/_tmp_bug_mechanism.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import bm25_index
import config
import hybrid_retriever as hr
import vector_index as vi

Q = "打印件表面有拉丝"
Q_MULTI = "打印件表面有拉丝，喷头移动时能拉出细丝"


def show(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def top5(r, sc, ix, k=5):
    sc, ix = np.asarray(sc).ravel(), np.asarray(ix).ravel()
    return [(r.chunks[int(ix[i])]["kb_id"], float(sc[i])) for i in range(k)]


show("实验 1：原判断「SVD n_components 越界抛」—— 只对了一半")
mini = None
for q in (Q, Q_MULTI):
    try:
        vb, m = vi._tfidf_svd_embed([q], fitted=None)
        print(f"  OK    {q[:22]!r}  vocab={len(m[0].vocabulary_)} "
              f"非零维={int((np.abs(vb) > 1e-9).sum())}")
        if mini is None:
            mini, vec_mini = m, vb
    except Exception as e:
        print(f"  CRASH {q[:22]!r}  {type(e).__name__}: {str(e)[:60]}")
print("  → 单词 query（TF-IDF 只切出 1 个 token → n_features=1 → actual_dim=0）**确实崩**；")
print("    多词 query（vocab=3 → actual_dim=2）**不崩**，返回迷你向量。原判断以偏概全。")

show("实验 2：更严重的一层 —— 跨索引空间污染（不崩、不报错、结果全错）")
r = hr.HybridRetriever()
f = vi.load_fitted_embedder(config.KB_FAISS_INDEX_PATH, strict=True)
print(f"  索引侧 vocab={len(f[0].vocabulary_)}")
if mini is not None:
    print(f"  迷你模型 vocab={len(mini[0].vocabulary_)}  ← 与索引毫无关系")
    sc_b, ix_b = r.vector_index.search(
        np.ascontiguousarray(vec_mini.astype("float32")), 5)
    print("  迷你向量查 kb 索引（维度都是 1024，FAISS 完全不报错）：")
    for kb, s in top5(r, sc_b, ix_b):
        print(f"    {kb}  cos={s:.4f}")
    print("  → 0.33x 这类分数**不是真实相似度**，是两个不同语义空间的内积。")

show("实验 3：修复后仍全零 —— 中文 query 对英文索引 TF-IDF 命中 0")
for q in [Q, "第一层没粘住", "Stringing"]:
    print(f"  nnz={f[0].transform([q]).nnz:4d}   {q}")
v = hr._embed_query(Q, fitted_path=r.faiss_path)
print(f"  查询向量非零维数 = {int((np.abs(v) > 1e-9).sum())}  → **全零向量**")
sc_z, ix_z = r.vector_index.search(np.zeros((1, 1024), dtype="float32"), 5)
print("  全零向量查索引得分全0，FAISS 按内部顺序返回：")
for kb, s in top5(r, sc_z, ix_z):
    print(f"    {kb}  cos={s:.4f}")
print("  根因：kb 索引语料是**英文**（KB001 text = 'Not Extruding at Start...'），")
print("  而 TfidfVectorizer 默认 token_pattern 不做中文分词，")
print("  中文 query 整串被当作 1 个 token → 词表命中 0。")

show("实验 4：端到端后果 —— 两路检索同时失效，返回字母序前 5 条")
out = r.retrieve(Q, top_k=5)
for x in out:
    print(f"    {x['kb_id']}  score={x['score']:.4f}  "
          f"bm25={x['bm25_score']:.4f}  cos={x['cosine']:.4f}")
print("  BM25 路：")
for idx, s in bm25_index.query_bm25(r.bm25, Q, top_k=3):
    print(f"    {r.chunks[idx]['kb_id']}  bm25={s:.4f}")
print("  → BM25 也被同一个 token_pattern 拖累（中文不分词），"
      "score 只剩 0.013 量级的位置先验。")

show("定稿结论")
print("  原 bug 机制 = **两条**，不是一条：")
print("   (a) 单词 query → TruncatedSVD n_components 越界**抛错**（吵闹，原判断对）")
print("   (b) 多词 query → 迷你模型向量查主索引，**静默返回无意义相似度**")
print("      （安静，更危险，与 assert_backend_consistent 拦的同类）")
print("  修复 bug 后暴露的**既有方法论缺陷**（与本 bug 无关，需另立任务）：")
print("   离线兜底（TF-IDF+SVD）**对中文 query 完全失效** —— 索引是英文、")
print("   分词器不做中文处理 → 双路皆零 → 返回 KB001~KB005 字母序。")
print("  建议：中文场景**必须配 DASHSCOPE_API_KEY**（生产本就如此），")
print("  或给 TfidfVectorizer 换中文分析器后重建索引。")
