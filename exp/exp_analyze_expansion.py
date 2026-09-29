"""分析 parent-child 扩展的 gold/噪声贡献，按 top-1 余弦置信度分箱，
定「门控扩展」的阈值：只在 top-1 高置信时扩展（补 gold 兄弟根因），
低置信（top-1 可能是错问题）时不扩展（避免灌噪声）。

不碰答案，只统计扩展块里 gold 占比，避免过拟合。
"""
import sys, json
sys.path.insert(0, "src")
from config import EVAL_SET_PATH
from hybrid_retriever import HybridRetriever
from skill_diagnose import DiagnoseSkill

ret = HybridRetriever()
skill = DiagnoseSkill()
es = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
inscope = [it for it in es if it["gold"]]

# 统计：完整扩展时，扩展块里 gold / 噪声 各多少；按 top-1 余弦分箱
from collections import defaultdict
bins = defaultdict(lambda: {"n":0, "gold_exp":0, "noise_exp":0, "has_gold_in_top5":0})
for it in inscope:
    q = it["query"]; gold = set(it["gold"])
    res = ret.retrieve(q, top_k=5)
    c0 = res[0]["cosine"]; c1 = res[1]["cosine"] if len(res)>1 else -1
    margin = c0 - c1
    top1_kb = res[0]["kb_id"]
    # 模拟扩展 top-1 的兄弟块
    exp = skill._expand_parent(top1_kb, {r["chunk_id"] for r in res})
    g_exp = sum(1 for e in exp if e["kb_id"] in gold)
    n_exp = len(exp) - g_exp
    top5_gold = any(r["kb_id"] in gold for r in res)
    # 分箱：按 c0
    b = "0.55-0.60" if c0<0.60 else "0.60-0.65" if c0<0.65 else "0.65-0.70" if c0<0.70 else "0.70-0.75" if c0<0.75 else "0.75+"
    d = bins[b]; d["n"]+=1
    d["gold_exp"]+=g_exp; d["noise_exp"]+=n_exp
    if top5_gold: d["has_gold_in_top5"]+=1

print("分箱(按 top-1 余弦 c0) | n | 扩展含gold块 | 扩展含噪声块 | top5本就有gold")
for b in sorted(bins):
    d = bins[b]
    print(f"  {b}: n={d['n']:3d} gold_exp={d['gold_exp']:3d} noise_exp={d['noise_exp']:3d} top5gold={d['has_gold_in_top5']:3d}")

# 门控模拟：仅当 c0 >= T 才扩展，统计
print("\n门控扩展模拟（仅 c0>=T 时扩展）：")
print("  T     | 会扩展的query | 其中扩展含gold(有益) | 其中扩展纯噪声(被砍) | 跳过扩展(可能丢gold兄弟)")
for T in (0.60, 0.65, 0.70, 0.75):
    exp_n=0; ben=0; noise_only=0; skip=0
    for it in inscope:
        q=it["query"]; gold=set(it["gold"])
        res=ret.retrieve(q, top_k=5); c0=res[0]["cosine"]; top1_kb=res[0]["kb_id"]
        exp=skill._expand_parent(top1_kb, {r["chunk_id"] for r in res})
        g_exp=sum(1 for e in exp if e["kb_id"] in gold); n_exp=len(exp)-g_exp
        if c0>=T:
            exp_n+=1
            if g_exp>0: ben+=1
            elif n_exp>0: noise_only+=1
        else:
            skip+=1
    print(f"  {T}: 扩展={exp_n:3d} 有益={ben:3d} 纯噪声={noise_only:3d} 跳过={skip:3d}")
