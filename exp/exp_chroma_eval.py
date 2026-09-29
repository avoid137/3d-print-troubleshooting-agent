"""Chroma 纯向量路 vs FAISS 向量路 · 等价性验证（gold 修正后口径）。

**目的**：证明把向量库从 FAISS 换成 Chroma 做「纯向量召回」时，
Recall@5 **零损失**——因为两者共用同一套 embedding，且向量 L2 归一化后
cosine 空间（Chroma `hnsw:space=cosine`）≡ 内积（FAISS `IndexFlatIP`）。

**两种 embedding 后端各验一次**：
  - tfidf    : TF-IDF + SVD 离线兜底（不调 API，可直接跑）
  - dashscope : text-embedding-v3（需 `DASHSCOPE_API_KEY`，缺则跳过）

**对每个后端**：
  1. 同进程生成 doc/query 向量 D、Q（与 chroma 内部**同一函数、同一 fit**）
  2. Chroma : `build_chroma` + `query_chroma` top-5 → 收集 kb_id
  3. FAISS  : `IndexFlatIP(D)` + `search(Q)` top-5 → 收集 kb_id
  4. 比 Recall@5（160 正例），逐类别 + 整体，断言两者**逐项相等**

注意：Chroma 对 dashscope 后端在 `hnsw:space=cosine` 下内部做归一化；
FAISS 侧对 dashscope 也显式 `l2norm` 后算 IP，两端都落在「真实余弦」上 → 排名一致。
tfidf 后端向量本就已归一化（`_tfidf_svd_embed` 内含），两端直接等价。

用法：
  python exp/exp_chroma_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from config import (CHUNKS_PATH, DASHSCOPE_API_KEY, TOP_K)          # noqa: E402
from chroma_index import build_chroma, query_chroma                 # noqa: E402
from vector_index import (_dashscope_embed, _tfidf_svd_embed)       # noqa: E402

CHUNKS = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
EVAL = json.loads((PROJ / "data" / "eval_set.json").read_text(encoding="utf-8"))
DOCS = [c["text"] for c in CHUNKS]
QUERIES = [e["query"] for e in EVAL]
CATS = sorted({e["category"] for e in EVAL})


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def make_vectors(backend):
    """返回 (D, Q)。backend ∈ {'tfidf','dashscope'}。"""
    if backend == "tfidf":
        D, fitted = _tfidf_svd_embed(DOCS)
        Q = _tfidf_svd_embed(QUERIES, fitted=fitted)
        return D, Q
    D = _dashscope_embed(DOCS)
    Q = _dashscope_embed(QUERIES)
    return D, Q


def chroma_recall(use_dashscope):
    """Chroma 纯向量路 Recall@5（不叠加 where 过滤）。"""
    build_chroma(use_dashscope=use_dashscope)   # 幂等（reset=True）
    hits = total = 0
    cat_hits, cat_total = {}, {}
    for e in EVAL:
        gold = set(e["gold"])
        if not gold:
            continue
        total += 1
        rs = query_chroma(e["query"], use_dashscope=use_dashscope, top_k=TOP_K)
        got = {r["kb_id"] for r in rs}
        cat_total[e["category"]] = cat_total.get(e["category"], 0) + 1
        if gold & got:
            hits += 1
            cat_hits[e["category"]] = cat_hits.get(e["category"], 0) + 1
    cats = {c: round(cat_hits.get(c, 0) / cat_total.get(c, 1), 4) for c in cat_total}
    return hits, total, cats


def faiss_recall(D, Q, top_k=TOP_K):
    """FAISS IndexFlatIP 纯向量路 Recall@5。D/Q 需已 L2 归一化。"""
    import faiss
    dim = D.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(D.astype(np.float32))
    hits = total = 0
    cat_hits, cat_total = {}, {}
    for i, e in enumerate(EVAL):
        gold = set(e["gold"])
        if not gold:
            continue
        total += 1
        _, idx = index.search(Q[i].astype(np.float32).reshape(1, -1), top_k)
        got = {CHUNKS[j]["kb_id"] for j in idx[0] if j != -1}
        cat_total[e["category"]] = cat_total.get(e["category"], 0) + 1
        if gold & got:
            hits += 1
            cat_hits[e["category"]] = cat_hits.get(e["category"], 0) + 1
    cats = {c: round(cat_hits.get(c, 0) / cat_total.get(c, 1), 4) for c in cat_total}
    return hits, total, cats


def fmt(h, t):
    return f"{h}/{t} = {h / t:.4f}"


def main():
    results = {}
    print("=" * 72)
    print("Chroma 纯向量路 vs FAISS 向量路 · 等价性验证")
    print("=" * 72)
    for backend in ("tfidf", "dashscope"):
        if backend == "dashscope" and not DASHSCOPE_API_KEY:
            print(f"\n[{backend}] 跳过：未配置 DASHSCOPE_API_KEY（纯向量召回无法用真实 embedding 验证）")
            results[backend] = {"skipped": True,
                                 "reason": "DASHSCOPE_API_KEY 未配置"}
            continue

        use_ds = (backend == "dashscope")
        print(f"\n[{backend}] 生成 embedding（{len(DOCS)} 条 doc / {len(QUERIES)} 条 query）...")
        D, Q = make_vectors(backend)

        h_c, t_c, cats_c = chroma_recall(use_ds)
        # FAISS 侧：dashscope 后端显式归一化（与 chroma cosine 对齐）；tfidf 已归一化
        Df = l2norm(D) if use_ds else D
        Qf = l2norm(Q) if use_ds else Q
        h_f, t_f, cats_f = faiss_recall(Df, Qf)

        identical = (h_c == h_f and t_c == t_f and cats_c == cats_f)
        print(f"  Chroma 向量路 Recall@5 : {fmt(h_c, t_c)}  | 各类别 {cats_c}")
        print(f"  FAISS  向量路 Recall@5 : {fmt(h_f, t_f)}  | 各类别 {cats_f}")
        print(f"  → 等价: {'✅ 逐项一致' if identical else '❌ 不一致!'}")

        results[backend] = {
            "skipped": False,
            "use_dashscope": use_ds,
            "chroma": {"hits": h_c, "total": t_c, "recall": round(h_c / t_c, 4), "by_category": cats_c},
            "faiss": {"hits": h_f, "total": t_f, "recall": round(h_f / t_f, 4), "by_category": cats_f},
            "identical": identical,
        }
        assert identical, f"{backend} 后端 chroma 与 faiss 向量召回不一致！"

    # 对照项目已发布的 FAISS 主口径（来自 README / 00_项目准备总览）
    print("\n" + "-" * 72)
    print("对照项目已发布口径（非本脚本产出，仅作参照）：")
    print("  vector(tfidf) : 117/160 = 0.7312  （FAISS 离线兜底）")
    print("  vector(dashscope): 120/160 = 0.7500 （FAISS 真实 embedding）")
    print("-" * 72)

    out = PROJ / "exp" / "chroma_eval_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")

    if all(r.get("identical", False) for r in results.values() if not r.get("skipped")):
        print("\n✅ 等价性验证通过：Chroma 纯向量召回 = FAISS 纯向量召回（零损失）")
    else:
        print("\n❌ 存在不一致，请检查")


if __name__ == "__main__":
    main()
