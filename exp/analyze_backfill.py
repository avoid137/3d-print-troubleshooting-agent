"""回填实验效果分析：读 exp_e2e_kb_backfill_result.json，产出
基线(v2.2一遍法) vs 处理(一遍法+回填) 的转化矩阵与按类别拆解。
"""
import json
from collections import Counter, defaultdict

ROOT = "D:/3D打印故障诊断Agent"
R = json.load(open(f"{ROOT}/exp/exp_e2e_kb_backfill_result.json", encoding="utf-8"))
recs = R["records"]

in_scope = [r for r in recs if r.get("in_scope") and not r.get("rejected", True)]
n = len(in_scope)

# 基线 / 处理 strict 计数
base_c = Counter(r.get("baseline_strict") for r in in_scope)
treat_c = Counter(r.get("strict") for r in in_scope)

# 转化矩阵（仅看 backfilled 题，最能说明回填作用）
conv = Counter()
for r in in_scope:
    if r.get("backfilled"):
        conv[(r.get("baseline_strict"), r.get("strict"))] += 1

# 按类别：基线 vs 处理 strict 成功数
by_cat_base = defaultdict(Counter)
by_cat_treat = defaultdict(Counter)
for r in in_scope:
    c = r.get("category", "?")
    by_cat_base[c][r.get("baseline_strict")] += 1
    by_cat_treat[c][r.get("strict")] += 1

# 回归（基线 correct -> 处理 非 correct）与 提升（非 correct -> correct）
regress = [r["id"] for r in in_scope if r.get("baseline_strict") == "correct" and r.get("strict") != "correct"]
improve = [r["id"] for r in in_scope if r.get("baseline_strict") != "correct" and r.get("strict") == "correct"]

base_correct = base_c["correct"]
treat_correct = treat_c["correct"]
base_rate = round(base_correct / n, 4)
treat_rate = round(treat_correct / n, 4)

print("=" * 60)
print(f"样本: in_scope 未拒答 = {n}")
print(f"基线 strict 成功 = {base_correct}/{n} = {base_rate*100:.2f}%  {dict(base_c)}")
print(f"处理 strict 成功 = {treat_correct}/{n} = {treat_rate*100:.2f}%  {dict(treat_c)}")
print(f"Δ strict = {(treat_rate-base_rate)*100:+.2f}pp  (净增 correct {treat_correct-base_correct})")
print("-" * 60)
print("回填题(p=%d) 转化矩阵 (基线->处理):" % sum(conv.values()))
for (b, t), k in sorted(conv.items(), key=lambda x: -x[1]):
    print(f"   {b:>8} -> {t:<8} : {k}")
print("-" * 60)
print("按类别 strict 成功数 (基线 -> 处理):")
for c in by_cat_base:
    bb = sum(1 for _ in by_cat_base[c] if True) and by_cat_base[c]
    tb = by_cat_treat[c]
    tot = sum(bb.values())
    print(f"   {c:<20} {bb.get('correct',0)}/{tot} -> {tb.get('correct',0)}/{tot}")
print("-" * 60)
print(f"回填提升(non-correct->correct) 题数 = {len(improve)}: {improve[:20]}")
print(f"回填回归(correct->非correct) 题数 = {len(regress)}: {regress}")
print("=" * 60)
