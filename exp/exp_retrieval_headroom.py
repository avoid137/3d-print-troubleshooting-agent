"""检索侧剩余空间离线探针（零 LLM 生成，仅 embedding）。

背景：B2 kb 级索引把「problem + intro + 全部 cause:solution（数千字）」拼成一整块，
症状信号被 solution 长文本稀释 → 端到端缺口分解显示 31/160(19.4%) 是"gold⊄top-5"漏召，
为单项最大缺口。本探针验证"症状信号加权/短文本向量"能否把召回拉起来。

评估指标（只对 in-scope 160 题）：
  - cov@5   : gold 全部出现在 top-5（严格，端到端 correct 的必要条件）
  - rec@5   : 至少命中 1 条 gold（宽松）
  - cov@10  : gold 全部出现在 top-10

变体（BM25 文本构造 × 向量文本构造）：
  full  = problem+intro+全部 cause:solution（基线 B2）
  sym   = problem+intro（纯症状，短）
  cause = problem+intro+仅 cause 短语（剔掉 solution 长文）
  w3    = problem 重复 3 次 + intro + 全部 cause:solution（词面加权，仅 BM25 侧）

用法：
    python exp/exp_retrieval_headroom.py
产物：exp/exp_retrieval_headroom_result.json + exp/retrieval_headroom_result.md
"""
import sys
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from config import KB_V2_PATH  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402

RESULT_PATH = ROOT / "exp" / "exp_retrieval_headroom_result.json"
VEC_CACHE = ROOT / "exp" / "_probe_vec_cache.pkl"
BM25_W, VEC_W, RRF_K = 0.6, 0.4, 60


# ---------------- 文本构造 ----------------
def build_texts():
    kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
    ids, full, sym, cause, w3 = [], [], [], [], []
    for it in kb:
        ids.append(it["kb_id"])
        prob = it["problem"]
        intro = (it.get("intro") or "").strip()
        head = prob + ("\n" + intro if intro else "")
        causes = it.get("causes", []) or []
        body_full = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution','')}"
                              for i, c in enumerate(causes))
        body_cause = "\n".join(f"{i+1}. {c['cause']}" for i, c in enumerate(causes))
        full.append(head + ("\n" + body_full if body_full else ""))
        sym.append(head)
        cause.append(head + ("\n" + body_cause if body_cause else ""))
        w3.append(" ".join([prob] * 3) + ("\n" + intro if intro else "")
                  + ("\n" + body_full if body_full else ""))
    return ids, {"full": full, "sym": sym, "cause": cause, "w3": w3}


# ---------------- 向量（带缓存，避免重复花钱） ----------------
def embed_cached(key, texts):
    cache = {}
    if VEC_CACHE.exists():
        cache = pickle.load(open(VEC_CACHE, "rb"))
    if key in cache and len(cache[key]) == len(texts):
        return cache[key]
    print(f"  [embed] {key}: {len(texts)} 条 ...", flush=True)
    v = _dashscope_embed(texts)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    cache[key] = v
    pickle.dump(cache, open(VEC_CACHE, "wb"))
    return v


# ---------------- 检索 ----------------
def rrf(bm25_top, vec_sim, ids, top_k):
    """bm25_top: [(idx, score)] 已排序；vec_sim: np.array(相似度)。"""
    bm_rank = {i: r for r, (i, _) in enumerate(bm25_top)}
    v_rank = {i: r for r, i in enumerate(np.argsort(-vec_sim))}
    fused = {}
    for i in set(bm_rank) | set(v_rank):
        s = 0.0
        if i in bm_rank:
            s += BM25_W / (RRF_K + bm_rank[i])
        if i in v_rank:
            s += VEC_W / (RRF_K + v_rank[i])
        fused[i] = s
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return [ids[i] for i, _ in ranked]


def evaluate(variant_name, bm_texts, vec_mat, qmat, items, ids):
    bm25 = BM25Okapi([tokenize(t) for t in bm_texts])
    n = len(items)
    c5 = c10 = r5 = 0
    gold_hit = 0.0
    for r in items:
        q = r["query"]
        top_bm = sorted(range(len(ids)),
                        key=lambda i: -bm25.get_scores(tokenize(q))[i])[:50]
        top_bm = [(i, 0.0) for i in top_bm]
        sim = vec_mat @ qmat[r["_qi"]]
        top5 = rrf(top_bm, sim, ids, 5)
        top10 = rrf(top_bm, sim, ids, 10)
        g = set(r["gold"])
        s5, s10 = set(top5), set(top10)
        if g <= s5:
            c5 += 1
        if g <= s10:
            c10 += 1
        if g & s5:
            r5 += 1
        gold_hit += len(g & s5) / len(g)
    return {"variant": variant_name, "cov@5": c5, "cov@5_rate": round(c5 / n, 4),
            "cov@10": c10, "cov@10_rate": round(c10 / n, 4),
            "rec@5": r5, "rec@5_rate": round(r5 / n, 4),
            "gold_hit_rate": round(gold_hit / n, 4), "n": n}


def main():
    ev = json.load(open(ROOT / "data" / "eval_set.json", encoding="utf-8"))
    prod = json.load(open(ROOT / "exp" / "exp_e2e_kb_backfill_prod_result.json",
                          encoding="utf-8"))
    in_scope = {r["id"] for r in prod["records"] if r.get("in_scope")}
    items = [r for r in ev if r["id"] in in_scope]
    print(f"in-scope 题目数: {len(items)}")

    ids, texts = build_texts()
    print(f"KB 条目数: {len(ids)}")

    # query 向量（算一次复用）
    qvecs = embed_cached("__queries__", [r["query"] for r in items])
    for i, r in enumerate(items):
        r["_qi"] = i

    mats = {k: embed_cached(f"kb_{k}", v) for k, v in texts.items()}

    variants = [
        ("A0 基线 BM25(full)+Vec(full)", texts["full"], mats["full"]),
        ("A1 BM25(w3)+Vec(full)",        texts["w3"],   mats["full"]),
        ("A2 BM25(full)+Vec(sym)",       texts["full"], mats["sym"]),
        ("A3 BM25(w3)+Vec(sym)",         texts["w3"],   mats["sym"]),
        ("A4 BM25(w3)+Vec(max(sym,full))", texts["w3"], np.maximum(mats["sym"], mats["full"])),
        ("A5 BM25(full)+Vec(max(sym,full))", texts["full"], np.maximum(mats["sym"], mats["full"])),
        ("A6 BM25(w3)+Vec(cause)",       texts["w3"],   mats["cause"]),
        ("A7 BM25(w3)+Vec(max(sym,cause))", texts["w3"], np.maximum(mats["sym"], mats["cause"])),
    ]
    results = []
    for name, bm_t, vm in variants:
        res = evaluate(name, bm_t, vm, qvecs, items, ids)
        results.append(res)
        print(f"  {name:32s} cov@5={res['cov@5']:3d}({res['cov@5_rate']:.2%}) "
              f"cov@10={res['cov@10']:3d}({res['cov@10_rate']:.2%}) "
              f"rec@5={res['rec@5']:3d}({res['rec@5_rate']:.2%}) "
              f"goldhit={res['gold_hit_rate']:.2%}", flush=True)

    json.dump({"n": len(items), "results": results},
              open(RESULT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
