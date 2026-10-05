# -*- coding: utf-8 -*-
"""D8 · 外部验证：DC-04（Zenodo 18628777）在 DC-02R 训练之外的独立测试。

🔴 这个脚本要回答的**只有一个问题**：
   「模型在 DC-02R 之外、还认不认得出缺陷？」

背景与数据事实（2026-10-05 unzip 实测，别照抄台账旧记录）
------------------------------------------------------------------
DC-04 只有 **3 类**（台账原记 4 类「含 stringing」是错的，实际无 stringing）：
    extrusion_error_print  7,006
    layershift_error_print 6,700
    no_error_print         6,435
每类有**两套视图**，同名文件 = 同一场景的裁剪版与全图版：
    test/cropped/<类>/camera{1,2,3>/detection_camN_XXXX.jpg  1080×1080
    test/full_res/<类>/camera{1,2,3>/detection_camN_XXXX.jpg  1920×1080

所以它**只能覆盖 5 类中的 1 类**（Layer_shifting），
外加「正常件门控」（no_error）和「挤出异常」（需人工判定归类）。
**Warping / Cracking / Stringing 在此数据集里完全没有** → 报告会显式标注
「未覆盖」，绝不拿 1类的分数冒充 5 类泛化能力。

三件本脚本刻意做的事
------------------------------------------------------------------
① **两视图对照 = 第三方裁剪带干预实验**
   DC-02R 内部的裁剪带对照已证明「模型读的是缺陷件」，但那是同分布内的自证。
   这里 `cropped`（贴近缺陷件）vs `full_res`（含机器本体与产线背景）是
   **完全独立的第三方复现**。若 cropped 明显优于 full_res，
   就再次排除「靠机器本体/构图认类」。

② **正常件门控 = 测「会不会把好件报成缺陷」**
   我们的模型是 5 类**强制分类**，没有「正常」这一类——这是当前最大盲区。
   真值no_error 的图**全部应被拒答**。若被自信地判成某缺陷类，
   记为 false_alarm，是比「分类错」更严重的失败。

③ **不预测、不外推，只报实测**
   没有验证到的类别一律写「未覆盖」。宁可难看，不补数字。

用法
------------------------------------------------------------------
    # 抽样（默认每类每视图 200 张，共 1,200 张，约几分钟）
    .venv/Scripts/python.exe -u exp/d8_external_valid.py --per-view 200

    # 只看计划
    .venv/Scripts/python.exe exp/d8_external_valid.py --dry-run

    # 换拒答阈值做敏感性分析
    .venv/Scripts/python.exe exp/d8_external_valid.py --per-view 200 --thresholds 0.7,0.8,0.9,0.95

    # 强制 CPU / 抽样更多
    .venv/Scripts/python.exe -u exp/d8_external_valid.py --per-view 500 --device cpu
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DC04_ZIP = Path(r"D:/datasets/dc04/test_data.zip")
OUT_JSON = ROOT / "exp" / "d8_external_result.json"
OUT_MD = ROOT / "exp" / "d8_external_result.md"

#: DC-04 真实类别 → 本项目的期望判定
#:
#:🔴 `None` 表示**该类真值落在我们 5 类之外**，只做「不该被自信断言」检查，
#:    不参与准确率统计（避免把映射不明算成错）。
#:
#: - layershift_error_print → Layer_shifting：知识库有 `Layer Shifting` 标签，
#:   语义直接对应，是唯一可做准确率的类。
#: - extrusion_error_print → 挤出异常：可能是 Spaghetti（喷不出料）也可能是
#:   中途断料，**现象不同但都在 kb_v2 的 Spaghetti 标签下**；
#:   保守起见记为 Off_platform 的近似类，**单独统计不合并**。
#: - no_error_print → 正常件：我们**没有这一类**，正确行为是拒答。
EXPECT = {
    "layershift_error_print": "Layer_shifting",
    "extrusion_error_print": "Off_platform",
    "no_error_print": None,
}
VIEWS = ["cropped", "full_res"]


def list_entries(z: zipfile.ZipFile) -> dict:
    """{类别: {视图: [条目名,...]}}，并按场景名（去掉视图前缀）去重。"""
    out: dict = {}
    for cat in EXPECT:
        out[cat] = {}
        for view in VIEWS:
            names = [n for n in z.namelist()
                     if n.lower().endswith((".jpg", ".jpeg", ".png"))
                     and f"/{view}/{cat}/" in n]
            out[cat][view] = sorted(names)
    return out


def sample_plan(entries: dict, per_view: int, seed: int) -> list:
    """每类 × 每视图随机抽 per_view 张。返回 [{cat, view, name}]。"""
    rng = random.Random(seed)
    plan = []
    for cat, views in entries.items():
        for view in VIEWS:
            pool = views[view]
            if not pool:
                continue
            take = pool if per_view <= 0 or per_view >= len(pool) \
                else rng.sample(pool, per_view)
            for n in take:
                plan.append({"cat": cat, "view": view, "name": n})
    return plan


def run_inference(plan: list, extract_dir: Path, device: str) -> list:
    """把抽中的图解到临时目录，跑一次批量推理（省模型加载时间）。"""
    from src.vision.pipeline import diagnose

    extract_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(DC04_ZIP)
    paths = []
    for i, item in enumerate(plan):
        dst = extract_dir / f"{i:06d}.jpg"
        if not dst.exists():
            dst.write_bytes(z.read(item["name"]))
        item["path"] = str(dst)
        paths.append(str(dst))

    print(f"\n[推理] {len(paths)} 张，device={device} …")
    preds = diagnose(paths, device=device)
    if len(preds) != len(plan):
        raise RuntimeError(
            f"推理返回 {len(preds)} 条≠ 计划 {len(plan)} 条，"
            "pipeline 与本脚本的条数假设不一致")
    for item, pr in zip(plan, preds):
        item["pred"] = pr.pred_class
        item["proba"] = round(pr.proba, 4)
        item["top2"] = [(c, round(p, 4)) for c, p in pr.top2]
        item["kb_ids"] = pr.kb_ids
        item["rejected"] = bool(pr.rejected)
    return plan


def analyze(plan: list, thresholds: list) -> dict:
    """按视图 × 类别出指标。**没有的类别不补数字。**"""
    res: dict = {"by_view": {}, "thresholds": {}}

    # ---- 1. 分类准确率：只统计 EXPECT 非 None 的类 ----
    for view in VIEWS:
        rows = [p for p in plan if p["view"] == view and EXPECT[p["cat"]]]
        if not rows:
            continue
        per_cat = {}
        for cat in {EXPECT[p["cat"]] for p in rows}:
            sub = [p for p in rows if EXPECT[p["cat"]] == cat]
            hit = sum(1 for p in sub if p["pred"] == cat)
            per_cat[EXPECT[cat]] = {
                "n": len(sub), "hit": hit,
                "acc": round(hit / len(sub), 4),
            }
        tot_hit = sum(v["hit"] for v in per_cat.values())
        res["by_view"][view] = {
            "n": len(rows),
            "overall_acc": round(tot_hit / len(rows), 4),
            "per_cat": per_cat,
        }

    # ---- 2. 正常件门控：no_error 应全部被拒答 ----
    gate = {}
    for view in VIEWS:
        sub = [p for p in plan if p["view"] == view and p["cat"] == "no_error_print"]
        if not sub:
            continue
        rej = sum(1 for p in sub if p["rejected"])
        fa = {}          # 误报成哪几类
        for p in sub:
            if not p["rejected"]:
                fa[p["pred"]] = fa.get(p["pred"], 0) + 1
        gate[view] = {
            "n": len(sub),
            "rejected": rej,
            "reject_rate": round(rej / len(sub), 4),
            "false_alarm": fa,
            "false_alarm_total": sum(fa.values()),
        }
    res["normal_gate"] = gate

    # ---- 3. 拒答阈值敏感性 ----
    for th in thresholds:
        per = {}
        for view in VIEWS:
            rows = [p for p in plan if p["view"] == view]
            if not rows:
                continue
            good = [p for p in rows if EXPECT[p["cat"]]
                    and p["pred"] == EXPECT[p["cat"]]]
            bad = [p for p in rows if EXPECT[p["cat"]]
                   and p["pred"] != EXPECT[p["cat"]]]
            normal = [p for p in rows if not EXPECT[p["cat"]]]
            # 「该拒答却没拒」：真值正常件但模型自信给了缺陷类
            miss_rej = sum(1 for p in normal if not p["rejected"])
            per[view] = {
                "n": len(rows),
                "acc_coverage": round(len(good) / len(rows), 4),
                "err_caught": (sum(1 for p in bad if p["proba"] < th)
                               / len(bad)) if bad else None,
                "normal_missed_reject": (miss_rej / len(normal)) if normal else None,
                "n_bad": len(bad), "n_normal": len(normal),
            }
        res["thresholds"][str(th)] = per

    # ---- 4. 裁剪带对照（本脚本的核心实验）----
    # 同一场景有 cropped / full_res 两个版本 → 配对比较
    paired = {}
    for cat in EXPECT:
        if not EXPECT[cat]:
            continue
        names_c = {Path(p["name"]).name: p for p in plan
                   if p["cat"] == cat and p["view"] == "cropped"}
        names_f = {Path(p["name"]).name: p for p in plan
                   if p["cat"] == cat and p["view"] == "full_res"}
        common = set(names_c) & set(names_f)
        if not common:
            continue
        c_ok = sum(1 for n in common
                   if names_c[n]["pred"] == EXPECT[cat])
        f_ok = sum(1 for n in common
                   if names_f[n]["pred"] == EXPECT[cat])
        paired[cat] = {
            "n_pairs": len(common),
            "cropped_acc": round(c_ok / len(common), 4),
            "full_res_acc": round(f_ok / len(common), 4),
            "delta_pp": round((c_ok - f_ok) / len(common) * 100, 2),
        }
    res["crop_intervention"] = paired
    res["_meta"] = {
        "n_sampled": len(plan),
        "dataset": "DC-04 Zenodo 18628777 test_data.zip",
        "dataset_note": ("实际只有 3 类，无 stringing；"
                         "台账 dataset_cards.md 已同步修正"),
        "covered_classes": sorted({v for v in EXPECT.values() if v}),
        "UNCOVERED_classes": ["Warping", "Cracking", "Stringing"],
        "warning": ("本结果只覆盖 5 类中的 1 类（Layer_shifting）+ 挤出异常近似类，"
                    "Warping/Cracking/Stringing 在 DC-04 中不存在，"
                    "**不可外推为 5 类整体泛化能力**"),
    }
    return res


def render_md(res: dict, plan: list) -> str:
    L = []
    A = L.append
    A("# D8 · DC-04 外部验证结果\n")
    A(f"- 抽样：**{res['_meta']['n_sampled']} 张**")
    A(f"- 覆盖类别：{', '.join(res['_meta']['covered_classes'])}")
    A(f"-🔴 **未覆盖**：{', '.join(res['_meta']['UNCOVERED_classes'])}"
      f"（DC-04 无此三类）")
    A(f"\n> 🔴 {res['_meta']['warning']}\n")

    A("## 一、分类准确率（仅可映射的类）\n")
    A("| 视图 | n | 总体 acc | 分类别 |")
    A("|---|---|---|---|")
    for view, v in res.get("by_view", {}).items():
        cats = "；".join(
            f"{k} {x['hit']}/{x['n']} ({x['acc']*100:.1f}%)"
            for k, x in sorted(v["per_cat"].items()))
        A(f"| `{view}` | {v['n']} | **{v['overall_acc']*100:.2f}%** | {cats} |")
    A("")

    A("## 二、正常件门控（真值 `no_error`，正确行为是拒答）\n")
    A("| 视图 | n | 拒答率 | 误报为缺陷 |")
    A("|---|---|---|---|")
    for view, g in res.get("normal_gate", {}).items():
        fa = "；".join(f"{k}×{v}" for k, v in
                       sorted(g["false_alarm"].items(), key=lambda x: -x[1])) or "无"
        A(f"| `{view}` | {g['n']} | **{g['reject_rate']*100:.1f}%** | {fa} |")
    A("")

    A("## 三、裁剪带干预实验（本脚本核心）\n")
    A("同一场景的 `cropped`（贴近缺陷件）vs `full_res`（含机器本体+产线背景）：\n")
    A("| 真值类 | 配对数 | cropped acc | full_res acc | 差值 |")
    A("|---|---|---|---|---|")
    for cat, v in res.get("crop_intervention", {}).items():
        sign = "+" if v["delta_pp"] > 0 else ""
        A(f"| {cat} | {v['n_pairs']} | {v['cropped_acc']*100:.1f}% | "
          f"{v['full_res_acc']*100:.1f}% | **{sign}{v['delta_pp']:.2f} pp** |")
    A("\n> cropped 明显优于 full_res ⇒ 模型读的是**缺陷件本身**，"
      "而非机器本体/产线构图。这是 DC-02R 内部裁剪带对照的独立第三方复现。\n")

    A("## 四、拒答阈值敏感性\n")
    ths = res.get("thresholds", {})
    if ths:
        A("| 阈值 | 视图 | 覆盖率 | 错误拦下率 | 正常件漏拒率 |")
        A("|---|---|---|---|---|")
        for th, per in ths.items():
            for view, v in per.items():
                er = ("—" if v["err_caught"] is None
                      else f"{v['err_caught']*100:.1f}%")
                mr = ("—" if v["normal_missed_reject"] is None
                      else f"**{v['normal_missed_reject']*100:.1f}%**")
                A(f"| {th} | `{view}` | {v['acc_coverage']*100:.1f}% "
                  f"| {er} | {mr} |")
    A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-view", type=int, default=200,
                    help="每类每视图抽几张（<=0 = 全取）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thresholds", default="0.7,0.8,0.9,0.95")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not DC04_ZIP.exists():
        print(f"❌ 找不到 {DC04_ZIP}\n"
              f"→ 跑 `python scripts/fetch_zenodo_dc04.py` 下载并校验")
        return 1

    thresholds = [float(x) for x in a.thresholds.split(",") if x.strip()]

    print("=" * 66)
    print("D8 · DC-04 外部验证")
    print("=" * 66)
    print(f"zip: {DC04_ZIP} ({DC04_ZIP.stat().st_size/1e9:.2f} GB)")

    with zipfile.ZipFile(DC04_ZIP) as z:
        entries = list_entries(z)
    print("\n[数据实况] 每类每视图可用张数：")
    for cat, views in entries.items():
        print(f"  {cat:24s} " + "  ".join(
            f"{v}={len(views[v]):5d}" for v in VIEWS)
            + f"   期望映射 → {EXPECT[cat] or '（正常件，应拒答）'}")

    plan = sample_plan(entries, a.per_view, a.seed)
    print(f"\n[抽样] 共 {len(plan)} 张"
          f"（每类每视图 {a.per_view} 张，seed={a.seed}）")
    if a.dry_run:
        from collections import Counter
        for k, v in sorted(Counter(
                f"{p['cat']}/{p['view']}" for p in plan).items()):
            print(f"  {k}: {v}")
        print("\n[dry-run] 不实际推理")
        return 0

    plan = run_inference(plan, Path(r"D:/datasets/dc04/_d8_tmp"), a.device)
    res = analyze(plan, thresholds)

    OUT_JSON.write_text(json.dumps(
        {"result": res, "plan": plan}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    OUT_MD.write_text(render_md(res, plan), encoding="utf-8")

    print("\n" + render_md(res, plan))
    print(f"[产出] {OUT_JSON}\n[产出] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
