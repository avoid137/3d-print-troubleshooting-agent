"""P1 · D3 结果统一汇总：把各L×骨干×口径的 json 汇成一张可对表的表。

为什么需要单独脚本（而不是手工填报告）：
    纪律要求「数字做完才填」+「跨材料口径一致」。手工抄数字是这两条纪律
    最常见的破口—— 一次抄错就再也看不出来。所以汇总必须**从 json 读**，
    并在本脚本里做一致性断言。

同时执行三条判定纪律：
    · 组内差异 < 2pp 一律判「等价」，不宣称优劣
    · Off_platform 仅 91 张（每折 18张），单折结果不可信，只报5 折×2 种子
    · 泄漏幅度 = stratified − grouped；|差| < 2pp 判「泄漏影响可忽略」

产物：exp/p1_d3_summary.json + 打印对��表
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
OUT = EXP / "p1_d3_summary.json"
EQUIV_PP = 2.0        # 小于此差值判「等价」
FILES = {
    ("L1", "efficientnet_b0"): "p1_d3_L1_efficientnet_b0.json",
    ("L2", "efficientnet_b0"): "p1_d3_L2_efficientnet_b0.json",
    ("L3", "efficientnet_b0"): "p1_d3_L3_efficientnet_b0.json",
    ("L1", "mobilenetv3_small_100"): "p1_d3_L1_mobilenetv3.json",
    ("L2", "mobilenetv3_small_100"): "p1_d3_L2_mobilenetv3.json",
    ("L3", "mobilenetv3_small_100"): "p1_d3_L3_mobilenetv3.json",
}


def unwrap(path: Path, level: str):
    """读单个 level 的结果，**结构无关地取**。

    ⚠️ 踩坑记录：`main()` 写出的结构是 `{backbone, img, levels:{L1:...}}`，
       早期版本写的是扁平 `{level, grouped:{...}}`。两种都在仓库里存在过。
       汇总必须兼容两者——**「漏读」比「读错」更可怕**：表格看起来是满的，
       实际少了几行，而读者无从察觉。
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    if "levels" in d:
        lv = d["levels"]
        if level in lv:
            return lv[level]
        if len(lv) == 1:
            return next(iter(lv.values()))
        return None
    if "grouped" in d and "macro_f1_mean" in d.get("grouped", {}):
        return d
    return None


def main():
    rows = []
    for (lv, bb), fn in FILES.items():
        p = EXP / fn
        if not p.exists():
            print(f"[skip] {fn} 未产出")
            continue
        d = unwrap(p, lv)
        if d is None:
            print(f"[skip] {fn} 结构无法识别（顶层键 {list(json.loads(p.read_text(encoding='utf-8')).keys())}）")
            continue
        g = d.get("grouped", {})
        s = d.get("stratified", {})
        if "macro_f1_mean" not in g:
            print(f"[skip] {fn} 无 grouped 结果")
            continue
        rows.append({
            "level": lv, "backbone": bb,
            "grouped_mean": g["macro_f1_mean"], "grouped_std": g["macro_f1_std"],
            "grouped_n": g["n_runs"],
            "strat_mean": s.get("macro_f1_mean"), "strat_std": s.get("macro_f1_std"),
            "leak_gap_pp": d.get("leakage_gap_pp"),
            "per_class": d.get("per_class_recall_mean", {}),
            "minutes": d.get("minutes"),
        })

    if not rows:
        print("没有任何 D3 结果可汇总")
        return 3

    print("\n" + "=" * 78)
    print("P1 · D3 三级对照汇总（EfficientNet-B0 / MobileNetV3-small，224px）")
    print("=" * 78)
    print(f"{'级别':<4} {'骨干':<24} {'grouped (主口径)':>20} {'stratified':>14} "
          f"{'泄漏pp':>8}")
    print("-" * 78)
    for r in rows:
        bb = "B0" if "efficientnet" in r["backbone"] else "MV3-small"
        g = f"{r['grouped_mean']*100:.2f} ± {r['grouped_std']*100:.2f}%"
        st = (f"{r['strat_mean']*100:.2f}%" if r["strat_mean"] is not None
              else "—")
        gap = (f"{r['leak_gap_pp']:+.2f}" if r["leak_gap_pp"] is not None else "—")
        print(f"{r['level']:<4} {bb:<24} {g:>20} {st:>14} {gap:>8}")

    # ---------------- 判定 ----------------
    verdicts = {}
    print("\n" + "-" * 78)
    print("判定")
    by_bb = {}
    for r in rows:
        by_bb.setdefault(r["backbone"], {})[r["level"]] = r

    for bb, lv in by_bb.items():
        name = "B0" if "efficientnet" in bb else "MV3-small"
        if "L1" in lv and "L2" in lv:
            d = abs(lv["L2"]["grouped_mean"] - lv["L1"]["grouped_mean"]) * 100
            eq = d < EQUIV_PP
            verdicts[f"{name}_L1_vs_L2"] = {
                "diff_pp": round(d, 2), "equivalent": eq,
                "verdict": ("等价（<2pp，不宣称优劣）" if eq
                            else f"L2 优于 L1 {d:.2f}pp")}
            print(f"  {name} L1 vs L2:差 {d:.2f}pp → "
                  f"{'等价，不宣称优劣' if eq else 'L2 更优'}")
        # 🔴 三级对照的核心问题：全量微调（L3）能不能超过冻结特征（L2）？
        #   小样本上微调常破坏预训练特征 → L3 可能**反而更低**。
        #   差值要连各自标准差一起看：L3 方差 10 倍于 L2 时，
        #   「L3 均值高0.1pp」完全落在噪声里。
        if "L2" in lv and "L3" in lv:
            dd = (lv["L3"]["grouped_mean"] - lv["L2"]["grouped_mean"]) * 100
            s2 = lv["L2"]["grouped_std"] * 100
            s3 = lv["L3"]["grouped_std"] * 100
            eq = abs(dd) < EQUIV_PP
            verdicts[f"{name}_L2_vs_L3"] = {
                "delta_pp": round(dd, 2),
                "std_L2": round(s2, 2), "std_L3": round(s3, 2),
                "equivalent": eq,
                "verdict": ("等价（|Δ|<2pp）——**全量微调未带来增益**"
                            if eq else
                            f"L3 {'优于' if dd > 0 else '低于'} L2 {abs(dd):.2f}pp"),
                "caveat": (f"注意 L3 标准差 {s3:.2f}pp 是 L2 的 {s3/max(s2,1e-9):.1f} 倍，"
                           f"均值差 {abs(dd):.2f}pp {'小于' if abs(dd) < s3 else '大于'}"
                           f"L3 自身波动 → 结论受折间方差支配。"),
            }
            print(f"  {name} L3 vs L2: Δ={dd:+.2f}pp"
                  f"（σ_L2={s2:.2f} σ_L3={s3:.2f}）→ "
                  f"{'等价，微调无增益' if eq else '有差异'}")
        for r in lv.values():
            if r["leak_gap_pp"] is not None:
                g = abs(r["leak_gap_pp"])
                key = f"{name}_{r['level']}_leak"
                verdicts[key] = {
                    "gap_pp": r["leak_gap_pp"],
                    "verdict": ("泄漏影响可忽略（|gap|<2pp）" if g < EQUIV_PP
                                else f"泄漏影响显著 {r['leak_gap_pp']:+.2f}pp")}
                print(f"  {name} {r['level']} 泄漏幅度 "
                      f"{r['leak_gap_pp']:+.2f}pp → "
                      f"{'可忽略' if g < EQUIV_PP else '显著'}")

    # 骨干间对照（同 level）
    b0, mv3 = by_bb.get("efficientnet_b0", {}), by_bb.get("mobilenetv3_small_100", {})
    for lv in ("L1", "L2", "L3"):
        if lv in b0 and lv in mv3:
            d = abs(b0[lv]["grouped_mean"] - mv3[lv]["grouped_mean"]) * 100
            eq = d < EQUIV_PP
            verdicts[f"backbone_{lv}"] = {
                "diff_pp": round(d, 2), "equivalent": eq,
                "verdict": "两骨干等价" if eq else "存在骨干差异"}
            print(f"  骨干对照 {lv}: B0 vs MV3 差 {d:.2f}pp → "
                  f"{'等价' if eq else '有差异'}")

    # ---------------- 一致性断言 ----------------
    print("\n" + "-" * 78)
    bad = []
    for r in rows:
        if r["grouped_n"] and r["grouped_n"] != 10:
            bad.append(f"{r['level']}/{r['backbone']} n_runs={r['grouped_n']} ≠ 10")
        if r["grouped_mean"] is not None and not (0.0 < r["grouped_mean"] <= 1.0):
            bad.append(f"{r['level']}/{r['backbone']} grouped_mean 越界")
    if bad:
        print("❌ 一致性问题:")
        for b in bad:
            print(f"   - {b}")
        return 1
    print("✅ 一致性断言通过（n_runs=10、macro-F1 ∈ (0,1]）")

    out = {
        "rows": rows, "verdicts": verdicts,
        "discipline": {
            "equiv_threshold_pp": EQUIV_PP,
            "main_scheme": "grouped",
            "n_runs_per_cell": 10,
            "note": ("Off_platform 仅 91 张（每折 18 张），单折结果不可信；"
                     "所有单元格均为 5 折 × 2 种子 = 10 次。"),
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
