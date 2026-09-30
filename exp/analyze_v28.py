"""v2.8 结果分析：救回 / 回归明细 + 类别拆解 + 目标集（73 道回填漏判）转化。

用法：python exp/analyze_v28.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("D:/3D打印故障诊断Agent")
RES = ROOT / "exp" / "exp_e2e_v28_result.json"
OUT = ROOT / "exp" / "e2e_v28_result.md"

res = json.load(open(RES, encoding="utf-8"))
recs = res["records"]
summary = res.get("summary", {})
inscope = [r for r in recs if r.get("in_scope") and not r.get("rejected", True)]

# 目标集：回填后仍非 correct
target = [r for r in inscope if r.get("backfill_strict") != "correct"]
target_conv = [r for r in target if r.get("v28_strict") == "correct"]
target_stay = [r for r in target if r.get("v28_strict") != "correct"]

improve = [r for r in inscope if r.get("backfill_strict") != "correct" and r.get("v28_strict") == "correct"]
regress = [r for r in inscope if r.get("backfill_strict") == "correct" and r.get("v28_strict") != "correct"]

# partial→correct 等细分
trans = Counter()
for r in target:
    trans[(r.get("backfill_strict"), r.get("v28_strict"))] += 1

lines = []
lines.append("# v2.8 枚举式结构化穷尽 —— 实验结果\n")
lines.append("## 总览（in-scope 共 %d 题）\n" % len(inscope))
lines.append("- 回填基线 correct 率： **%.2f%%** (%s)" % (
    summary["backfill_rate"] * 100, summary["backfill_strict"]))
lines.append("- v2.8 correct 率： **%.2f%%** (%s)" % (
    summary["v28_rate"] * 100, summary["v28_strict"]))
lines.append("- 净 Δpp： **%+.2f pp**" % summary["delta_pp"])
lines.append("- 救回数（回填非 correct → v28 correct）： **%d**" % len(improve))
lines.append("- 回归数（回填 correct → v28 非 correct）： **%d**" % len(regress))
lines.append("- lenient：回填 %.2f%% → v2.8 %.2f%%" % (
    summary["backfill_lenient_rate"] * 100, summary["v28_lenient_rate"] * 100))

lines.append("\n## 目标集（回填后仍非 correct 的 %d 题）\n" % len(target))
lines.append("- 转 correct： **%d** （救回）" % len(target_conv))
lines.append("- 仍非 correct： %d" % len(target_stay))
lines.append("\n转化矩阵（回填 → v2.8）：")
for (bf, v28), c in sorted(trans.items(), key=lambda x: -x[1]):
    lines.append("-  %s → %s : %d" % (bf, v28, c))

lines.append("\n## 类别拆解\n")
lines.append("| 类别 | n | 回填正确 | v2.8正确 | 回填率 | v2.8率 | Δpp |")
lines.append("|---|---|---|---|---|---|---|")
for cat, d in summary.get("by_category", {}).items():
    dpp = round((d["v28_rate"] - d["bf_rate"]) * 100, 2)
    lines.append("| %s | %d | %d | %d | %.2f%% | %.2f%% | %+0.2f |" % (
        cat, d["n"], d["bf_correct"], d["v28_correct"],
        d["bf_rate"] * 100, d["v28_rate"] * 100, dpp))

lines.append("\n## 救回明细（%d 题）\n" % len(improve))
for r in improve:
    lines.append("- `%s` [%s] 回填=%s → v2.8=correct｜%s" % (
        r["id"], r["category"], r.get("backfill_strict"),
        r.get("query", "")[:60]))

if regress:
    lines.append("\n## ⚠️ 回归明细（%d 题，须为零）\n" % len(regress))
    for r in regress:
        lines.append("- `%s` [%s] 回填=correct → v2.8=%s｜%s" % (
            r["id"], r["category"], r.get("v28_strict"),
            r.get("query", "")[:60]))
else:
    lines.append("\n## ✅ 零回归（回填 correct 题全部保持 correct）\n")

# 目标集仍 non-correct：看 skipped 是否把 gold 跳了
lines.append("\n## 目标集仍 non-correct 的样例（前 20，含 skipped 供归因）\n")
for r in target_stay[:20]:
    sk = r.get("v28_skipped", [])
    lines.append("- `%s` [%s] 回填=%s v2.8=%s skip=%s｜%s" % (
        r["id"], r["category"], r.get("backfill_strict"), r.get("v28_strict"),
        sk, r.get("query", "")[:50]))

md = "\n".join(lines)
OUT.write_text(md, encoding="utf-8")
print(md)
print("\n已写出", OUT)
