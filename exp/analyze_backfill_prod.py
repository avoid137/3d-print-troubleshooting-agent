"""生产级回填（集成 src）结果分析：读 exp_e2e_kb_backfill_prod_result.json。
同 analyze_backfill.py，但 baseline/treatment 均由集成代码在同一次运行内算出。
"""
import json
from collections import Counter, defaultdict

ROOT = "D:/3D打印故障诊断Agent"
R = json.load(open(f"{ROOT}/exp/exp_e2e_kb_backfill_prod_result.json", encoding="utf-8"))
recs = R["records"]
if "summary" in R:
    print("=== summary ===")
    print(json.dumps(R["summary"], ensure_ascii=False, indent=2))

in_scope = [r for r in recs if r.get("in_scope") and not r.get("rejected", True)]
n = len(in_scope)
base_c = Counter(r.get("baseline_strict") for r in in_scope)
treat_c = Counter(r.get("strict") for r in in_scope)

conv = Counter()
for r in in_scope:
    if r.get("backfilled"):
        conv[(r.get("baseline_strict"), r.get("strict"))] += 1

by_cat_base = defaultdict(Counter)
by_cat_treat = defaultdict(Counter)
for r in in_scope:
    c = r.get("category", "?")
    by_cat_base[c][r.get("baseline_strict")] += 1
    by_cat_treat[c][r.get("strict")] += 1

regress = [r["id"] for r in in_scope if r.get("baseline_strict") == "correct" and r.get("strict") != "correct"]
improve = [r["id"] for r in in_scope if r.get("baseline_strict") != "correct" and r.get("strict") == "correct"]

print("=" * 60)
print(f"in_scope 未拒答 = {n}")
print(f"baseline strict = {base_c['correct']}/{n} = {round(base_c['correct']/n,4)*100:.2f}%  {dict(base_c)}")
print(f"treatment strict = {treat_c['correct']}/{n} = {round(treat_c['correct']/n,4)*100:.2f}%  {dict(treat_c)}")
print(f"Δ = {(treat_c['correct']-base_c['correct'])/n*100:+.2f}pp  (净增 {treat_c['correct']-base_c['correct']})")
print(f"correct 回归 = {len(regress)}  {regress}")
print(f"correct 提升 = {len(improve)}  {improve[:20]}")
print("-" * 60)
print("回填题转化矩阵 (基线->处理):")
for (b, t), k in sorted(conv.items(), key=lambda x: -x[1]):
    print(f"   {b:>8} -> {t:<8} : {k}")
print("-" * 60)
print("按类别 strict 成功数 (基线 -> 处理):")
for c in by_cat_base:
    bb = by_cat_base[c]; tb = by_cat_treat[c]; tot = sum(bb.values())
    print(f"   {c:<20} {bb.get('correct',0)}/{tot} -> {tb.get('correct',0)}/{tot}")
print("=" * 60)
