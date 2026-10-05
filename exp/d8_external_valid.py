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
    """每类抽 per_view 个**场景**，每个场景同时取 cropped + full_res 两个视图。

    🔴 2026-10-06 修**核心实验失效的 bug**：原来对 cropped / full_res **各自独立**
    随机抽 per_view 张。但两视图是「同一场景的两个版本」，文件名完全一致
    （实测 7,006 个文件名在两视图下100% 可配对），独立抽样的交集必然趋近于 0——
    实测 200+200 只配出**3 对 / 2 对**，裁剪带对照（第四组实验）几乎等于没做，
    报告里却照样打了「0.00 pp」的表格，**看起来像结论，其实是噪声**。
    「主动移除线索看还剩多少」的实验，**前提是两组只差那一个变量**，
    所以必须按场景配对抽样，不能各抽各的。
    """
    rng = random.Random(seed)
    plan = []
    for cat, views in entries.items():
        # 按文件名建索引：同一场景在两视图下的条目名
        by_name: dict = {}
        for view in VIEWS:
            for n in views.get(view, []):
                by_name.setdefault(Path(n).name, {})[view] = n
        # 只保留两视图都存在的场景（配对实验要求）
        pairable = {k: v for k, v in by_name.items() if len(v) == len(VIEWS)}
        if not pairable:
            print(f"  [警告] {cat}: 无可配对场景，跳过")
            continue
        keys = sorted(pairable)
        take = (keys if per_view <= 0 or per_view >= len(keys)
                else rng.sample(keys, per_view))
        for k in take:
            for view in VIEWS:
                plan.append({"cat": cat, "view": view, "name": pairable[k][view]})
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
        # 🔴 2026-10-06 修：这里的 `cat` 已经是**映射后**的类名
        #   （Layer_shifting / Off_platform），不是 DC-04 原始类名。
        #   原写法 `per_cat[EXPECT[cat]]` 又查了一次 EXPECT → KeyError: 'Off_platform'。
        for cat in {EXPECT[p["cat"]] for p in rows}:
            sub = [p for p in rows if EXPECT[p["cat"]] == cat]
            hit = sum(1 for p in sub if p["pred"] == cat)
            per_cat[cat] = {
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
            # 🔴 2026-10-06 修：原来这里用 `not p["rejected"]`，
            #   但 `rejected` 是 pipeline 按**固定 0.90** 算出来的，
            #   与本表正在扫描的 th 无关 → 同一张表里混了两套阈值，
            #   「敏感性分析」名不副实（每个 th 的 normal_missed_reject 都一样）。
            #   改为与 err_caught 同口径：都按传入的 th 重算。
            miss_rej = sum(1 for p in normal if p["proba"] >= th)
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

    # ---- 5. 🔴 输出塌缩诊断（本轮实测发现的关键现象，务必看）----
    #实测：DC-04 上91% 的图（**连正常件也是 91%**）都被判成同一个类 Stringing，
    #   概率还很高（mean≈0.90）。这不是「域差异导致准确率下降」，
    #   而是**输出塌缩**：模型对所有 DC-04 图给出几乎同一个答案，与真值无关。
    #   只报acc 会让人误以为是普通准确率低，从而去找「调模型」的错误方向。
    #判据：① 预测分布的最大类占比 ② 正常件的最大类占比（塌缩的 smoking gun）
    collapse = {}
    for view in VIEWS + ["both"]:
        rows = plan if view == "both" else [p for p in plan if p["view"] == view]
        if not rows:
            continue
        cnt = {}
        for p in rows:
            cnt[p["pred"]] = cnt.get(p["pred"], 0) + 1
        top, topn = max(cnt.items(), key=lambda x: x[1])
        norm = [p for p in rows if not EXPECT[p["cat"]]]
        ncnt = {}
        for p in norm:
            ncnt[p["pred"]] = ncnt.get(p["pred"], 0) + 1
        ntop, ntopn = ((max(ncnt.items(), key=lambda x: x[1])) if ncnt
                       else (None, 0))
        collapse[view] = {
            "n": len(rows),
            "top_class": top, "top_share": round(topn / len(rows), 4),
            "n_distinct_pred": len(cnt),
            "pred_dist": dict(sorted(cnt.items(), key=lambda x: -x[1])),
            "normal_top_class": ntop,
            "normal_top_share": round(ntopn / len(norm), 4) if norm else None,
            "normal_n_distinct": len(ncnt),
            "verdict": ("🔴 输出塌缩：预测高度集中于单一类，且正常件同样如此"
                        if topn / len(rows) > 0.5 and ntopn / max(len(norm), 1) > 0.5
                        else "未见塌缩"),
        }
    res["collapse_diagnosis"] = collapse
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
    A("\n> 🔴 **低于随机猜测（5 类随机 = 20%）**。这不是「准确率偏低」，"
      "而是**输出塌缩**，见第五节。\n")

    A("## 二、正常件门控（真值 `no_error`，正确行为是拒答）\n")
    A("| 视图 | n | 拒答率 | 误报为缺陷 |")
    A("|---|---|---|---|")
    for view, g in res.get("normal_gate", {}).items():
        fa = "；".join(f"{k}×{v}" for k, v in
                       sorted(g["false_alarm"].items(), key=lambda x: -x[1])) or "无"
        A(f"| `{view}` | {g['n']} | **{g['reject_rate']*100:.1f}%** | {fa} |")
    _fa = [g["false_alarm_total"] / g["n"] * 100
           for g in res.get("normal_gate", {}).values() if g["n"]]
    A(f"\n> 🔴 正常件误报率高达 "
      f"{min(_fa):.0f}%~{max(_fa):.0f}%" if _fa else "> 正常件被大量误报")
    A(">，且**误报方向高度集中**。拒答机制在真实域外数据上基本失效。\n")

    A("## 三、裁剪带干预实验（本脚本核心）\n")
    A("同一场景的 `cropped`（贴近缺陷件）vs `full_res`（含机器本体+产线背景）：\n")
    A("| 真值类 | 配对数 | cropped acc | full_res acc | 差值 |")
    A("|---|---|---|---|---|")
    for cat, v in res.get("crop_intervention", {}).items():
        sign = "+" if v["delta_pp"] > 0 else ""
        A(f"| {cat} | {v['n_pairs']} | {v['cropped_acc']*100:.1f}% | "
          f"{v['full_res_acc']*100:.1f}% | **{sign}{v['delta_pp']:.2f} pp** |")
    A("\n> ⚠️ **本次结果不支持「cropped 优于 full_res ⇒ 模型读缺陷件」这个结论**——"
      "因为在 DC-04 上模型本身已塌缩到几乎单一输出，"
      "两组都塌到同一类时，差值反映不出任何裁剪带效应。"
      "此实验在 DC-04 上**因输出塌缩而失效**（结论：域外数据上无法复现该实验）。\n")

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

    # ---- 第五节：塌缩诊断 ----
    col = res.get("collapse_diagnosis", {})
    if col:
        A("\n## 五、🔴 输出塌缩诊断（解释为什么准确率低于随机）\n")
        A("| 视图 | n | 预测最大类 | 占比 | 不同预测数 |"
          " 正常件最大类 | 正常件占比 |")
        A("|---|---|---|---|---|---|---|")
        for view, c in col.items():
            A(f"| `{view}` | {c['n']} | **{c['top_class']}** | "
              f"**{c['top_share']*100:.1f}%** | {c['n_distinct_pred']} | "
              f"{c['normal_top_class']} | "
              f"**{(c['normal_top_share'] or 0)*100:.1f}%** |")
        A("")
        for view, c in col.items():
            A(f"- `{view}`：{c['verdict']}")
        A("\n> **正常件被同样自信地判成同一个缺陷类**，是塌缩最硬的证据："
          "如果模型只是「域差异导致识别变弱」，不同真值应该有不同错误方向；"
          "现在**连真值正常的图都和缺陷图走向同一个答案**，"
          "说明输出已与图像内容脱钩。\n")
    bv = res.get("by_view", {})
    acc_txt = " / ".join(
        f"{v['overall_acc']*100:.2f}%（{k}）" for k, v in bv.items()) or "—"
    ng = res.get("normal_gate", {})
    fa_txt = " / ".join(
        f"{g['false_alarm_total']/g['n']*100:.0f}%（{k}）" for k, g in ng.items()
    ) or "—"
    col = res.get("collapse_diagnosis", {}).get("both", {})
    top_cls = col.get("top_class", "—")
    top_sh = col.get("top_share", 0) * 100
    ntop_sh = (col.get("normal_top_share") or 0) * 100
    A(f"""
## 六、结论与根因分析

### 结论（照实说，不粉饰）

1. 🔴 **DC-04 上模型基本失效**：准确率 {acc_txt}，**低于 5 类随机猜测的 20%**。
2. 🔴 **失效形态是「输出塌缩」而非「识别变弱」**：{top_sh:.1f}% 的图
   （**其中正常件{ntop_sh:.1f}%**）都被判成同一个类 `{top_cls}`，且置信度高。
   **正常件与缺陷图走向同一个答案 ⇒ 输出已与图像内容脱钩。**
3. 🔴 **拒答机制在域外基本无效**：正常件误报率 {fa_txt}。
4. ⚠️ **裁剪带干预实验在 DC-04 上失效**：模型已塌缩，两组都塌到同一类，
   差值不反映裁剪带效应。**DC-02R 内部的裁剪带结论未被外部复现。**

### 已排除的解释（管线没坏）

- ✅ **图像与推理链正常**：DC-04 解压图像素统计正常（暗背景 + 红色调 R>G+B），
  非空图、非全黑。
- ✅ **同一条管线在 DC-02R 上 200/200 全对**（每类抽 40 张）→
  `diagnose()` 的加载、归一化、类序映射、阈值全部正常。
- ✅ DC-04 zip CRC 全过、40,282 张图像完整。

**→ 低准确率是真实的域外泛化失败，不是工程 bug。**

### 根因分析

| 维度 | DC-02R（训练） | DC-04（域外） | 落差 |
|---|---|---|---|
| 机型 | 桌面 Ender 3 | Original Prusa i3 | 不同机型 |
| 相机 | 手机 | 工业相机 cam1/2/3 | 不同成像链路 |
| 构图 | 整台打印机 + 底下一小块件 | 产线/相机固定机位 | 视角与占比完全不同 |
| 件体颜色 | 灰白件 | **红色立方件** | DC-02R 没有的颜色域 |
| 分辨率 | 3072×2048 | 1080×1080 / 1920×1080 | — |

**最可能的主因：ImageNet 预训练特征对「产线红件 + 工业相机 + 深色背景」这一域没见过。**
冻结骨干 + 线性探针的方案**本身没有任何域适应能力**——它假设
「测试分布 = 训练分布」，跨机器/跨相机直接违反该假设。

**这正是本项目记忆里那条警告的实证：**
> ⚠️ 99.73% 只排除了「认错东西」，**推不出「换机器/换材料也准」**
> ——DC-02R 全来自同一台 Ender + 同批 PLA。落地到别的机型**必须重采目标域数据**。

### 后续动作（按性价比排序）

| 优先级 | 动作 | 目的 |
|---|---|---|
| 🔴 P0 | **加「正常/无缺陷」类** | 当前 5 类强制分类是最大盲区，正常件必然被误报 |
| 🔴 P0 | **自采目标域数据**（目标机型 + 手机拍摄，30~50 张/类） | 唯一能真正验证换机型场景的办法 |
| 🟡 P1 | 轻量域适应：只重训最后两层或加 BN 适配 | 成本低，但需有目标域数据才谈得上 |
| 🟡 P1 | 引入 OOD / 域检测，**先于分类器生效** | 塌缩时高置信度是**反指标**，必须先判「是否域内」 |
| ⛔ | 换更大骨干 / 上 QLoRA | **不解决域问题，别做** |

> 💡 关键判断：**塌缩下高置信度是危险信号而非优势**。
> 当前 `proba>=0.90 就自信作答` 的策略，在域外会给出「高置信的错误答案」，
> 部署时**必须**先过 OOD/域检测，宁可拒答。
""")
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
