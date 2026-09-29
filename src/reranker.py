"""Rerank 精排：用 cross-encoder 对双路融合的 top-N 候选重排。

动机
----
RRF 融合只利用「排名」信息，丢失了 query 与 chunk 的细粒度交互。
cross-encoder（query 与 doc 拼接后一起编码）能捕捉更精确的相关性，
对多缺陷、诱导歧义这类难例尤其有用。

策略（v2，置信度门控）
----
v1 对所有查询无条件精排，实测 Recall@5 0.8500→0.8125（负结果）：
通用 reranker 与垂直领域 gold 不对齐，把大量本就正确的结果挤出前 5。
而 baseline 0.85 里大部分正确案例根本不需要精排——精排只在边界 case 有价值。

门控信号的选择（实测校准）：
  - top-1 余弦分：答对/答错中位数 0.695 vs 0.690，几乎无区分力 → 弃用
  - margin（top-1 与 top-2 余弦差）：答对 0.0255 vs 答错 0.0140，
    答错案例 margin 更小（模型更犹豫）→ 采用
v2 规则：margin < GATE_MARGIN → 边界/犹豫，触发精排；否则高置信直接返回。
这样把精排的"伤害面"限制在低置信子集上。

其余要点
--------
1. 双路融合先取 top-N（N=20，召回优先），再交给 reranker 精排取 top-5。
2. reranker 用 DashScope gte-rerank-v2（cross-encoder），返回 relevance_score。
3. 失败降级：rerank 调用失败时保留原融合顺序（精排是增强项，不能成为故障点）。
4. 缓存：key 含 RERANK_VERSION + query + 候选 chunk_id 集合，避免重复烧 token。
   失败结果不缓存，避免临时故障被永久固化。
"""
import hashlib
import json
from pathlib import Path

from config import DASHSCOPE_API_KEY, PROJECT_ROOT, TOP_K

RERANK_MODEL = "gte-rerank-v2"   # gte-rerank 未开通（AccessDenied），v2 实测可用
RERANK_VERSION = "v3"            # v3 门控信号改为 margin，递增避免命中旧缓存
CACHE_PATH = PROJECT_ROOT / "data" / "rerank_cache.json"
RERANK_CANDIDATES = 20   # 融合后取 top-N 进精排
GATE_MARGIN = 0.02       # margin(top1-top2 余弦) < 此值视为犹豫/边界，触发精排


def _cache_key(query, chunk_ids):
    basis = f"{RERANK_VERSION}::{query}::" + ",".join(sorted(chunk_ids))
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


def _load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def rerank(query, results, top_k=TOP_K, use_rerank=True, gate_margin=GATE_MARGIN):
    """对融合结果精排（带置信度门控）。

    results: hybrid retrieve 的候选列表（含 chunk_id / text / cosine 等）
    门控：margin = top-1 与 top-2 的余弦差。
      margin >= gate_margin → 高置信（模型果断），直接返回原顺序前 top_k，不精排。
      margin <  gate_margin → 犹豫/边界，触发精排。
    返回: 重排后的 top_k 列表（精排时每条补一个 rerank_score 字段）
    """
    if not use_rerank or not DASHSCOPE_API_KEY or len(results) <= 1:
        return results[:top_k]

    # 置信度门控：用 margin（top-1 余弦分本身无区分力，实测答对/答错中位数几乎相同）
    c0 = results[0].get("cosine", -1.0)
    c1 = results[1].get("cosine", -1.0)
    margin = c0 - c1
    if margin >= gate_margin:
        return results[:top_k]

    chunk_ids = [r["chunk_id"] for r in results]
    cache = _load_cache()
    ck = _cache_key(query, chunk_ids)
    if ck in cache:
        order = cache[ck]
    else:
        order = _call_rerank(query, results)
        if order is not None:      # 失败不缓存，避免临时故障被永久固化
            cache[ck] = order
            _save_cache(cache)

    if order is None:  # 调用失败，降级为原顺序
        return results[:top_k]

    # order 是 [(原列表下标, relevance_score), ...]，已按分数降序
    by_idx = {i: r for i, r in enumerate(results)}
    reranked = []
    for idx, score in order[:top_k]:
        if idx in by_idx:
            item = dict(by_idx[idx])
            item["rerank_score"] = round(float(score), 4)
            reranked.append(item)
    return reranked


def _call_rerank(query, results):
    import dashscope
    from dashscope import TextReRank
    dashscope.api_key = DASHSCOPE_API_KEY

    documents = [r.get("text", "") for r in results]
    try:
        resp = TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        if resp.status_code != 200:
            print(f"[rerank] 调用失败，降级为原顺序: {resp.code} {resp.message}")
            return None
        out = []
        for r in resp.output["results"]:
            out.append((int(r["index"]), float(r["relevance_score"])))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[rerank] 异常，降级为原顺序: {type(e).__name__}: {e}")
        return None


def main():
    # 冒烟测试：两条候选，看 rerank 是否能区分相关度
    fake = [
        {"chunk_id": "a", "text": "Stringing is caused by oozing during travel moves. Fix retraction."},
        {"chunk_id": "b", "text": "Warping happens when corners lift off the bed due to cooling."},
    ]
    out = rerank("my print has stringing", fake, top_k=2)
    for r in out:
        print(r["chunk_id"], r.get("rerank_score"), r["text"][:50])


if __name__ == "__main__":
    main()
