"""留出集（held-out）范围判定验证。

为什么要做
----------
本项目的判官 prompt 是在 `data/eval_set.json` 的 200 条上**迭代出来的**
（v2 → v7，每一版都看同一份评测集）。这带来一个方法论风险：

    在同一份评测集上反复调 prompt，会**过拟合**到这份集子；
    F1 0.9873 很可能只是「在这 200 条上」的上限，不代表泛化能力。

本脚本用**全新手写**的 40 条 query（不参与任何 prompt 调整）独立验证。
标签只分二元：`in_scope`（应放行）/ `out_of_scope`（应拒答）。

⚠️ 诚实说明本验证的局限
----------------------
这 40 条是**在 prompt 定稿之后**由同一作者写的，所以它只能证明
「规则不是只对原来 40 条负例生效」，**不能完全排除作者偏见**。
真正严格的留出验证需要**第三方标注**或**线上真实 query 日志**。

用法
----
    python exp_holdout_scope.py            # 留出集 #1（40 条）
    python exp_holdout_scope.py --set 2    # 留出集 #2（24 条，验证 v8）
    python exp_holdout_scope.py --reuse    # 从已存结果重算
"""
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import PROJECT_ROOT                      # noqa: E402
from hybrid_retriever import HybridRetriever         # noqa: E402
from rejector import judge_many, PROMPT_VERSION, JUDGE_MODEL   # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "holdout_scope_result.json"

# (query, 期望)  期望: True=库内(应放行) / False=库外(应拒答)
CASES = [
    # ---------- 库内：现象式（10）----------
    ("my print has fuzzy blobs on the surface", True),
    ("first layer looks squished and wavy", True),
    ("corners of my part curl up after a few layers", True),
    ("there are gaps between walls on the top surface", True),
    ("my print is weak and snaps easily along layer lines", True),
    ("the top of my overhang droops badly", True),
    ("extruder clicking and gaps in the print", True),
    ("elephant foot on the bottom of my prints", True),
    ("one side of my print is rough like sandpaper", True),
    ("supports are fusing to the part and hard to remove", True),
    # ---------- 库内：参数/校准式（6）----------
    ("what flow rate should I use for matte PLA", True),
    ("how much part cooling fan for PLA bridges", True),
    ("retraction settings to reduce stringing on a direct drive", True),
    ("my temperature tower looks best at 205, what should I change", True),
    ("z hop height to avoid knocking over tall parts", True),
    ("should I lower acceleration to reduce ghosting", True),
    # ---------- 库内：机械根因（2）----------
    ("could loose belts cause shifted layers", True),
    ("z rod wobble causing banding on the walls", True),
    # ---------- 库内：口语化 / 否定条件（2）----------
    ("no clog but extrusion keeps stopping mid print", True),
    ("warping is fine, it is the layer adhesion that fails", True),

    # ---------- 库外：选购/选型（3）----------
    ("which budget printer is best for beginners", False),
    ("should I buy a glass bed or a PEI sheet", False),
    ("what filament brand has the most consistent diameter", False),
    # ---------- 库外：切片软件功能操作（3）----------
    ("how do I rotate a model in Cura", False),
    ("how to install a slicer plugin", False),
    ("where is the settings menu in PrusaSlicer", False),
    # ---------- 库外：教程/参考资料（3）----------
    ("3D printing glossary of terms", False),
    ("beginner tutorial for FDM printing", False),
    ("dimensional tolerance chart for FDM", False),
    # ---------- 库外：硬件改装/接线/固件（3）----------
    ("how to replace the hotend on my Ender 3", False),
    ("which firmware should I flash on my printer", False),
    ("wiring the bltouch sensor to the board", False),
    # ---------- 库外：非 FDM 工艺（3）----------
    ("cleaning the resin tank after a failed print", False),
    ("what is DLP printing used for", False),
    ("how to cure resin prints with UV light", False),
    # ---------- 库外：科普 / 商业 / 后处理（5）----------
    ("history of 3D printing technology", False),
    ("how does FDM printing work in general", False),
    ("how to start a 3D printing service business", False),
    ("how to sand and prime a print for painting", False),
    ("is 3D printed plastic recyclable at home", False),
]

# 留出集 #2：用于验证 prompt v8。
# ⚠️ 方法论说明：#1 的失败样本已被用于修正 prompt，故 #1 不再是「无偏」留出集；
#    验证修正效果必须用**另写**的一份。#2 特意覆盖 #1 未涉及的形态
#    （对比表 / 规格数据 / 择优 / 固件 / 材料处理）。
CASES_2 = [
    # ---------- 库内（12）----------
    ("layers are separating when I bend the part", True),
    ("thin strings webbing between the towers", True),
    ("the nozzle drags and leaves marks on the top layer", True),
    ("the print warps only on the front edge near the fan", True),
    ("my extrusion looks uneven, some lines come out thin", True),
    ("small features come out melty and deformed", True),
    ("the seam line is very visible on a round object", True),
    ("bridges sag in the middle", True),
    ("should I reduce layer height to get finer detail", True),
    ("what bed temperature stops ABS from lifting", True),
    ("could a loose extruder gear cause under extrusion", True),
    ("the print starts curling after about 20 layers", True),
    # ---------- 库外（12）----------
    ("which slicer is the most popular", False),
    ("how to add a camera to octoprint", False),
    ("FDM vs SLS comparison table", False),
    ("PLA material properties datasheet", False),
    ("how to design snap fit joints", False),
    ("upgrading to a direct drive extruder", False),
    ("how to sell my 3D printed models online", False),
    ("nozzle size comparison chart", False),
    ("what is the strongest infill pattern", False),
    ("how to dry filament in an oven", False),
    ("best 3D scanner under 500 dollars", False),
    ("how to install klipper firmware", False),
]


def summarize(rows, set_no=1):
    """rows: [{query, expect, got, reason}]"""
    pos = [r for r in rows if r["expect"]]      # 库内
    neg = [r for r in rows if not r["expect"]]  # 库外

    pos_ok = sum(1 for r in pos if r["got"])          # 正确放行
    pos_bad = [r for r in pos if not r["got"]]        # 误拒
    neg_ok = sum(1 for r in neg if not r["got"])      # 正确拒答
    neg_bad = [r for r in neg if r["got"]]            # 漏拒

    acc = (pos_ok + neg_ok) / len(rows)
    out = {
        "set": set_no,
        "prompt_version": PROMPT_VERSION,
        "judge_model": JUDGE_MODEL,
        "n": len(rows),
        "accuracy": round(acc, 4),
        "in_scope_total": len(pos),
        "in_scope_passed": pos_ok,
        "in_scope_false_reject": round(len(pos_bad) / len(pos), 4) if pos else None,
        "out_scope_total": len(neg),
        "out_scope_rejected": neg_ok,
        "out_scope_leak": round(len(neg_bad) / len(neg), 4) if neg else None,
        "false_rejects": [{"query": r["query"], "reason": r["reason"]} for r in pos_bad],
        "leaks": [{"query": r["query"], "reason": r["reason"]} for r in neg_bad],
    }

    print(f"\n{'=' * 70}")
    print(f"留出集 #{set_no} · {len(rows)} 条（prompt {PROMPT_VERSION}, {JUDGE_MODEL}）")
    print("=" * 70)
    print(f"  总体准确率     : {pos_ok + neg_ok}/{len(rows)} = {acc:.2%}")
    print(f"  库内 {len(pos)} 条 → 正确放行 {pos_ok}/{len(pos)}"
          f" = {pos_ok / len(pos):.2%}   误拒 {len(pos_bad)} 条")
    print(f"  库外 {len(neg)} 条 → 正确拒答 {neg_ok}/{len(neg)}"
          f" = {neg_ok / len(neg):.2%}   漏拒 {len(neg_bad)} 条")
    if pos_bad:
        print("\n  ❌ 误拒（规则可能过紧）:")
        for r in pos_bad:
            print(f"     - {r['query']}\n       {r['reason'][:80]}")
    if neg_bad:
        print("\n  ❌ 漏拒（规则可能过松）:")
        for r in neg_bad:
            print(f"     - {r['query']}\n       {r['reason'][:80]}")
    if not pos_bad and not neg_bad:
        print("\n  ✅ 40 条全部判定正确")
    return out


def main():
    reuse = "--reuse" in sys.argv
    set_no = 1
    if "--set" in sys.argv:
        set_no = int(sys.argv[sys.argv.index("--set") + 1])
    cases = CASES if set_no == 1 else CASES_2
    out_path = (PROJECT_ROOT / "data" /
                ("holdout_scope_result.json" if set_no == 1
                 else f"holdout_scope_result{set_no}.json"))

    if reuse and out_path.exists():
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[--reuse] 从 {out_path.name} 读取，仅重算。")
        summarize(saved["rows"])
        return

    r = HybridRetriever()
    print(f"留出集 #{set_no}：检索 + 判官 {len(cases)} 条 ...")
    t0 = time.time()
    pairs = [(q, r.retrieve(q, top_k=5)) for q, _ in cases]
    verdicts = judge_many(pairs)
    rows = []
    for (q, expect), v in zip(cases, verdicts):
        rows.append({
            "query": q,
            "expect": expect,
            "got": bool(v["answerable"]),
            "reason": v.get("reason", ""),
        })
    out = summarize(rows, set_no)
    out_path.write_text(json.dumps({"rows": rows, **out}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n耗时 {time.time() - t0:.1f}s ｜ 明细已存 {out_path}")


if __name__ == "__main__":
    main()
