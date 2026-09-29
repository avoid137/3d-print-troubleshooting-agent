"""分析 rerank 对 gold 召回与精度的影响（不碰答案，只为定阈值）。

对每条库内评测 query：retrieve top-5 → 强制 rerank（gate_margin=999）
统计：
  1) RRF 原始 top-5 的 gold@5（kb_id 级）
  2) rerank 后 top-5 的 gold@5
  3) rerank 后 top-3 的 gold@3（若改用 top-3 喂生成器，gold 丢多少）
  4) 在 reranked top-5 中，gold 块 vs 无关块的 rerank_score 分布
     ——据此判断“丢弃低分块”能否只去噪声、不伤 gold
"""
import sys, json
sys.path.insert(0, "src")
from config import EVAL_SET_PATH, TOP_K
from hybrid_retriever import HybridRetriever
from reranker import rerank

ret = HybridRetriever()

es = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
inscope = [it for it in es if it["gold"]]

gold_at5_raw = 0
gold_at5_re = 0
gold_at3_re = 0
gold_total = 0
gold_in_top5_count = 0          # gold kb 出现在 rerank top-5 的条数（按 query）
re_gold_scores = []             # reranked top-5 中 gold 块的 rerank_score
re_nongold_scores = []          # reranked top-5 中 无关块的 rerank_score
dropped_gold_if_top3 = 0        # 若只留 rerank top-3，会丢 gold 的 query 数

for it in inscope:
    q = it["query"]
    gold = set(it["gold"])
    gold_total += len(gold)
    res = ret.retrieve(q, top_k=TOP_K)
    # 原始 RRF top-5 gold 命中
    raw_kb = {r["kb_id"] for r in res}
    if gold & raw_kb:
        gold_at5_raw += 1
    # 强制 rerank
    reranked = rerank(q, res, top_k=TOP_K, gate_margin=999)
    re_kb_top5 = {r["kb_id"] for r in reranked}
    if gold & re_kb_top5:
        gold_at5_re += 1
        gold_in_top5_count += 1
    re_kb_top3 = {r["kb_id"] for r in reranked[:3]}
    if not (gold & re_kb_top3):
        dropped_gold_if_top3 += 1
    # 分数分布
    for i, r in enumerate(reranked):
        sc = r.get("rerank_score", None)
        if sc is None:
            continue
        if r["kb_id"] in gold:
            re_gold_scores.append(sc)
        else:
            re_nongold_scores.append(sc)

n = len(inscope)
def pct(x): return f"{x}/{n} = {100*x/n:.1f}%"

print(f"库内 query 数: {n}")
print(f"[召回] RRF 原始 gold@5      : {pct(gold_at5_raw)}")
print(f"[召回] rerank 后 gold@5     : {pct(gold_at5_re)}  (丢 {n-gold_at5_re})")
print(f"[召回] rerank 后 gold@3     : 若只留 top-3，会丢 gold 的 query = {pct(dropped_gold_if_top3)}")
print()
import statistics as st
def desc(name, arr):
    if not arr:
        print(f"{name}: (空)"); return
    arr_s = sorted(arr)
    print(f"{name}: n={len(arr)} min={min(arr):.3f} p25={arr_s[len(arr)//4]:.3f} "
          f"median={st.median(arr):.3f} p75={arr_s[3*len(arr)//4]:.3f} max={max(arr):.3f}")
desc("rerank_score(gold 块, top-5 内)", re_gold_scores)
desc("rerank_score(无关块, top-5 内)", re_nongold_scores)
# 建议阈值：若以“无关块 max 分”为界，gold 会被误丢多少
if re_nongold_scores and re_gold_scores:
    nongold_max = max(re_nongold_scores)
    ng_med = st.median(re_nongold_scores)
    gold_min = min(re_gold_scores)
    print()
    print(f"无关块 max={nongold_max:.3f}, median={ng_med:.3f}; gold 块 min={gold_min:.3f}")
    # 若只保留 rerank_score >= floor 的块（且始终保留 top-1）
    for floor in (0.2, 0.3, 0.4, 0.5):
        keep_gold = sum(1 for s in re_gold_scores if s >= floor)
        drop_gold = len(re_gold_scores) - keep_gold
        keep_nongold = sum(1 for s in re_nongold_scores if s >= floor)
        drop_nongold = len(re_nongold_scores) - keep_nongold
        print(f"  floor={floor}: 保留 gold 块 {keep_gold}/{len(re_gold_scores)} "
              f"(丢 {drop_gold}) | 保留 无关块 {keep_nongold}/{len(re_nongold_scores)} "
              f"(去噪 {drop_nongold})")
