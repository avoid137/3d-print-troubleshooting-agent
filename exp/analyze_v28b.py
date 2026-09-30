"""v2.8b 结果分析：v2.8 草稿 + 回填保底，对比回填基线 / v2.8 基线。

用法：python exp/analyze_v28b.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("D:/3D打印故障诊断Agent")
RES = ROOT / "exp" / "exp_e2e_v28b_result.json"
OUT = ROOT / "exp" / "e2e_v28b_result.md"

v = json.load(open(RES, encoding="utf-8"))
s = v["summary"]
recs = v["records"]
inscope = [r for r in recs if r.get("in_scope") and not r.get("rejected", True)]

reg_bf = [r for r in inscope if r.get("backfill_strict") == "correct" and r.get("v28b_strict") != "correct"]
imp_bf = [r for r in inscope if r.get("backfill_strict") != "correct" and r.get("v28b_strict") == "correct"]
reg_v28 = [r for r in inscope if r.get("v28_strict") == "correct" and r.get("v28b_strict") != "correct"]
imp_v28 = [r for r in inscope if r.get("v28_strict") != "correct" and r.get("v28b_strict") == "correct"]

L = []
L.append("# v2.8b 混合（v2.8 草稿 + append-only 回填保底）—— 实验结果\n")
L.append("## 总览（in-scope 共 %d 题）\n" % s["n_in_scope"])
L.append("- 回填基线 correct 率： **%.2f%%** (%s)" % (s["backfill_rate"]*100, s["backfill_strict"]))
L.append("- v2.8 单独 correct 率： **%.2f%%** (%s)" % (s["v28_rate"]*100, s["v28_strict"]))
L.append("- **v2.8b correct 率： %.2f%% (%s)  = v2.8 完全等价**" % (s["v28b_rate"]*100, s["v28b_strict"]))
L.append("- Δ vs 回填： **%+.2f pp** ｜ Δ vs v2.8： %+.2f pp" % (s["delta_vs_backfill_pp"], s["delta_vs_v28_pp"]))
L.append("- 回归 vs 回填（须=0）： **%d** ← 安全门未过（与 v2.8 相同 23 道）" % s["regress_vs_backfill"])
L.append("- 救回 vs 回填： %d" % s["improve_vs_backfill"])
L.append("- 回归 vs v2.8（回填须零新增）： **%d** ✅ ｜ 救回 vs v2.8： %d" % (s["regress_vs_v28"], s["improve_vs_v28"]))
L.append("- lenient：回填 %.2f%% ／ v2.8 %.2f%% ／ v2.8b %.2f%%" % (s["backfill_lenient_rate"]*100, s["v28_lenient_rate"]*100, s["v28b_lenient_rate"]*100))
L.append("\n## 核心结论")
L.append("- v2.8b **严格等价于 v2.8**（91/160，23 回归原封不动）。")
L.append("- 回填保底**零新增回归但也零回收**：v2.8 的 skipped 是**显式拒绝**根因（答案写\"未采纳 KBxxx\"），")
L.append("  回填只补\"默默漏掉\"的相关根因、不推翻主动拒绝 → 混合方案救不了 skip 回归。")
L.append("- 故 v2.8 系列（含混合）**无法在零回归前提下超过回填 SOTA 54.37%**。")
L.append("\n## 类别拆解（回填 / v2.8 / v2.8b）\n")
L.append("| 类别 | n | 回填正确 | v2.8正确 | v2.8b正确 | 回填率 | v2.8率 | v2.8b率 |")
L.append("|---|---|---|---|---|---|---|---|")
for c, d in s["by_category"].items():
    L.append("| %s | %d | %d | %d | %d | %.2f%% | %.2f%% | %.2f%% |" % (
        c, d["n"], d["bf_correct"], d["v28_correct"], d["v28b_correct"],
        d["bf_rate"]*100, d["v28_rate"]*100, d["v28b_rate"]*100))
L.append("\n## 回归 vs 回填（%d 道，与 v2.8 完全相同）\n" % s["regress_vs_backfill"])
for r in reg_bf:
    L.append("- `%s` [%s] 回填=correct → v2.8b=%s｜%s" % (r["id"], r["category"], r.get("v28b_strict"), r.get("query","")[:50]))
L.append("\n## 救回 vs 回填（%d 道，来自 v2.8 的 27 道增益，回填保底未增减）\n" % s["improve_vs_backfill"])
for r in imp_bf:
    L.append("- `%s` [%s] 回填=%s → v2.8b=correct｜%s" % (r["id"], r["category"], r.get("backfill_strict"), r.get("query","")[:50]))

md = "\n".join(L)
OUT.write_text(md, encoding="utf-8")
print(md)
print("\n已写出", OUT)
