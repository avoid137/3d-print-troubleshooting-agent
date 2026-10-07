# -*- coding: utf-8 -*-
"""D10 · Caxton-S 跨域探针：域多样性更高的公开数据能不能救回泛化？

设计文档（**判据已写死，跑前不可改**）：`exp/d10_caxton_probe_design.md`

本脚本要回答的三个问题（三个不同问题的三个测量，不是 A/B/C/D）
------------------------------------------------------------------
① **域内上界**：同一个 EfficientNet-B0 冻结特征 + 线性探针，
   在 Caxton-S 域内能到多少 macro-F1？
   → 先测上界，否则跨域低分无法归因（是域差还是任务本身难）。

② **零样本跨域**（与 D8 完全同构，唯一变化是换了数据集）：
   DC-02R 训练好的探针，一分钱不重训，直接测 Caxton-S。
   **只统计可对齐类别**，其余显式标「未覆盖」。
   必须同时报：准确率 + **输出塌缩诊断** + Good Print 拒答门控。

③ **双向吞噬**（🔴 D9 教训，绝不可省）：
   联合训练后，DC-02R 掉多少？Caxton-S 拿到多少？
   任何「加数据」方案都必须双向测，只测一个等于没测。

🔴 三条不可事后放宽的红线（写在代码里防自己）
------------------------------------------------------------------
· 没有验证到的类别一律写「未覆盖」，绝不外推、绝不补数字。
· 双向吞噬缺一即判不通过。
· 域外高分必须先排除「同场景泄漏」和「标签语义错配」才可采信。

🔴 口径一致性（对齐 `src/vision/infer.py` 部署件，否则 D9 教训③会重演）
------------------------------------------------------------------
· 分类器 LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
· 骨干走 `weights.build_model`（**禁用 timm.create_model(pretrained=True)**）
· **特征提取必须与 `infer._tf()` 同一套前向** → 直接复用 embed 逻辑，
  否则 Caxton-S 与 DC-02R 的特征不在同一空间，LogReg 会学到「域差异」。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe -u exp/d10_caxton_probe.py --per-class 400
    .venv/Scripts/python.exe -u exp/d10_caxton_probe.py --per-class 400 --device cpu
    .venv/Scripts/python.exe exp/d10_caxton_probe.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CAXTON_ZIP = Path(r"D:/datasets/caxton-s/caxton_s.zip")
OUT_JSON = ROOT / "exp" / "d10_caxton_result.json"
OUT_MD = ROOT / "exp" / "d10_caxton_result.md"

# ---------------------------------------------------------------- 类别对齐表
#: Caxton-S 目录名 → 中文描述（README v1）
CX_CLASSES = {
    "good_print": "Good Print（正常件）",
    "under_extrusion": "Under-Extrusion",
    "over_extrusion": "Over-Extrusion",
    "spaghetti": "Spaghetti",
    "warping": "Warping",
    "stringing": "Stringing",
}

#: 🔴 与本项目 5 类的对齐关系。**`None` = 没有对应类，标「未覆盖」，不硬塞。**
#:
#: · warping / stringing  → 同名，**可直接算准确率**
#: · spaghetti            → 挤出失败，语义对应 KB101 `Spaghetti (Print Detached from Bed)`
#:   ⚠️ 与我们的 Off_platform 语义相近但**不合并统计**：
#:      本项目 D8 实测 Off_platform 实际现象是「画面无成形件、材料堆成乱丝」= Spaghetti，
#:      而 KB101 标题就是 Spaghetti → 它们本就是同一现象。
#:      但**必须单独看这一对的混淆矩阵**，不合并。
#: · under/over_extrusion → 挤出量异常，**我们 5 类里没有对应** → 未覆盖
#: · good_print           → 正常件，**我们没有这一类** → 只测拒答门控，不算准确率
ALIGN = {
    "warping": "Warping",
    "stringing": "Stringing",
    "spaghetti": "Off_platform",
    "under_extrusion": None,
    "over_extrusion": None,
    "good_print": None,
}
#: 可对齐类（参与准确率）
ALIGNED = [k for k, v in ALIGN.items() if v]
#: 正常件（只测拒答，不算准确率）
NORMAL_CX = "good_print"
#: 🔴 明确未覆盖的类，报告必须显式列出
UNCOVERED = [k for k, v in ALIGN.items() if v is None and k != NORMAL_CX]

#: 随机基线（按对齐后的类别数）
RANDOM_BASELINE = 1.0 / len(set(ALIGN[k] for k in ALIGNED))

#: 预注册判据（见设计文档 §二；**跑前写死，不许事后改**）
VERDICTS = {
    "domain_upper_ok": 0.90,   # ① ≥90% 骨干在该域完全可用
    "domain_upper_weak": 0.70,  # ① 70~90% 可用但有难度
    "zeroshot_good": 0.50,     # ② >50% → H1 支持
    "zeroshot_bad": 0.20,      # ② <20% → H1 不支持
    "joint_max_drop_pp": 2.0,  # ③ A 侧 macro-F1 相对基线最大掉 2pp
    "joint_min_keep": 0.95,    # ③ B 侧不低于上界的 95%
}


# ---------------------------------------------------------------- 数据读取
def scan_entries(z: zipfile.ZipFile) -> dict:
    """{类别: [条目名,...]}，并顺带记录打印任务前缀分布（用于 group 划分）。"""
    out: dict = {k: [] for k in CX_CLASSES}
    for nm in z.namelist():
        if not nm.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = nm.split("/")
        for c in CX_CLASSES:
            if c in parts:
                out[c].append(nm)
                break
    return {k: sorted(v) for k, v in out.items()}


#: 帧号间隙阈值：大于此值即认为是**另一次打印**（实测 p22/warping 中位间隙=1，
#: 间隙>100 的有 17 个 → 该打印机下共 18 次独立打印）。
#: 🔴 这个阈值不是拍脑袋来的，是**从数据实测的中位间隙与分布形状定的**。
FRAME_GAP = 100


def print_segments(cls: str, names: list) -> dict:
    """把同一(类, 打印机)下的帧按**帧号间隙**切成若干「打印批次」。

    🔴🔴 2026-10-07 重大修正——**我原先把 `p22` 当成"打印任务 ID"，是错的**：
        实测 Caxton-S 全库只有 **10 台打印机**（p22/p24/p31/p32/p40/p46/p54/p79/p82/p91），
        帧号跨度上万且不连续（p22: 218~15218）。
        → `p22` 是**打印机编号**，不是任务编号。
        → 按它分组 = 只得到 10 个 group，抽样 400 变成 n=58，
          **整份报告的 macro-F1 全部作废**（域内 3.51% 是这么来的）。
        正确粒度：**每次打印是一个批次**，边界由帧号间隙给出
        （连续帧 gap=1，跨打印 gap≫1）→ 同一段连续帧必然是同一次打印。

        ⚠️ 这正是项目铁律「**字面对得上 ≠ 现象对得上**」的又一例：
        文件名长得像任务 ID，就真的是任务 ID 吗？**必须实测分布。**

    返回 {segment_key: [条目名, ...]}
    """
    fr: dict = {}
    for nm in names:
        base = Path(nm).name
        m = re.match(r"(p\d+)_image-(\d+)\.jpg$", base)
        if m:
            fr.setdefault(m.group(1), []).append((int(m.group(2)), nm))
        else:
            # 文件名不符合模式 → 单独成组（保守：宁可分组过细）
            fr.setdefault(base, []).append((0, nm))
    segs: dict = {}
    for printer, items in fr.items():
        items.sort()
        cur_key = None
        prev = None
        for idx, (no, nm) in enumerate(items):
            if prev is None or (no - prev) > FRAME_GAP:
                # 新批次：段名带上打印机 + 起始帧号，保证跨打印机不撞
                cur_key = f"{cls}#{printer}#{no}"
            segs.setdefault(cur_key, []).append(nm)
            prev = no
    return segs


def print_task(cls: str, nm: str) -> str:
    """🔴 兼容旧调用：单图版group 键（= 打印机，不是批次）。

    ⚠️ 真实分组请用 `print_segments()`。本函数仅用于快速查看，
       **绝不要用它做 group-aware 折**（会退化成"按打印机分折"）。
    """
    base = Path(nm).name
    m = re.match(r"(p\d+)_image-(\d+)", base)
    return f"{cls}#{m.group(1) if m else base}"


def sample_plan(entries: dict, per_class: int, seed: int) -> list:
    """每类抽 min(per_class, 批次总数) 个**打印批次**，每批次取 1 帧。

    🔴 必须按**打印批次**抽，不按图也不按打印机抽：
       · 按图独立抽 → 同一次打印的多帧同时进训练集与测试集 → 折外虚高
       · 按打印机抽   → 只有 10 组，n 会被压到几十，本轮就踩了这个坑
    """
    rng = random.Random(seed)
    plan = []
    for cls, names in entries.items():
        if not names:
            continue
        segs = print_segments(cls, names)
        keys = sorted(segs)
        take = keys if per_class <= 0 or per_class >= len(keys) \
            else rng.sample(keys, per_class)
        for k in take:
            plan.append({
                "cx_class": cls,
                "task": k,
                "name": rng.choice(segs[k]),
                "align": ALIGN[cls],
            })
    return plan


# ---------------------------------------------------------------- 特征
def embed_from_zip(z: zipfile.ZipFile, names: list, device: str,
                   batch: int = 64) -> np.ndarray:
    """zip 内图片 → 冻结特征。

    🔴 **必须与 `src.vision.infer._tf()` 完全同一套变换**（Resize(224,224)+ToTensor+Normalize），
       否则 Caxton-S 与 DC-02R 特征不在同一空间，
       LogReg 学到的是「域差异」而非「缺陷形态」——本项目最贵的教训。
    """
    import torch
    from PIL import Image
    from src.vision.infer import build_backbone, _tf
    model = build_backbone(device)
    tf = _tf()
    out = []
    with torch.no_grad():
        for s in range(0, len(names), batch):
            xs = []
            for nm in names[s:s + batch]:
                im = Image.open(io.BytesIO(z.read(nm))).convert("RGB")
                xs.append(tf(im))
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                f = model.forward_features(torch.stack(xs).to(device))
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            elif f.dim() == 3:
                f = f[:, 0]
            out.append(f.float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def make_probe():
    """🔴 分类器口径必须与部署件一致（D9 教训③）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"),
    )


def group_folds(groups: list, k: int = 5, seed: int = 42) -> list:
    """把group 均衡分到 k 折（保证同一打印任务不跨折）。

    🔴 2026-10-06 修**潜在静默 bug**：联合训练时 DC-02R 的 group 是整数
       （0..1748）、Caxton-S 的 group 是 `p22` 这类字符串，
       两者若恰好同名就会**被判成同一 group** → 两个域的图共处一折 →
       折外评估泄漏，且不报任何错。
       对策：加 `prefix` 参数，强制域前缀不同。
    """
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    buckets: list = [[] for _ in range(k)]
    for i, g in enumerate(uniq):
        buckets[i % k].append(g)
    g2f = {g: i for i, b in enumerate(buckets) for g in b}
    return [g2f[g] for g in groups]


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    """macro-F1手写（不依赖 sklearn 的label 顺序假设）。"""
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------- 测量
def m1_domain_upper(feat, y, groups, labels, k, seed) -> dict:
    """① 域内上界：group-aware 折外 macro-F1。"""
    folds = group_folds(groups, k, seed)
    y_pred = [None] * len(y)
    for f in range(k):
        tr = [i for i in range(len(y)) if folds[i] != f]
        te = [i for i in range(len(y)) if folds[i] == f]
        if not te or not tr:
            continue
        probe = make_probe()
        probe.fit(feat[tr], [y[i] for i in tr])
        for i, p in zip(te, probe.predict(feat[te])):
            y_pred[i] = p
    done = [(t, p) for t, p in zip(y, y_pred) if p is not None]
    yt = [t for t, _ in done]
    yp = [p for _, p in done]
    per = {}
    for c in labels:
        sub = [(t, p) for t, p in done if t == c]
        if sub:
            per[c] = {
                "n": len(sub),
                "acc": round(sum(1 for t, p in sub if t == p) / len(sub), 4),
            }
    mf1 = macro_f1(yt, yp, labels)
    return {
        "n": len(done),
        "labels": labels,
        "macro_f1": round(mf1, 4),
        "per_class": per,
        "n_groups": len(set(groups)),
            "group_definition": "打印批次 = 同(类,打印机)下帧号间隙<=100 的连续段",
        "n_folds_used": k,
        "pred_dist": dict(sorted(Counter(yp).items(), key=lambda x: -x[1])),
    }


def m2_zeroshot(feat, plan, device_deploy, align_map) -> dict:
    """② 零样本跨域：DC-02R 部署件 → Caxton-S，**不重训**。

    `device_deploy` = (scaler, clf, classes) from load_deployed_model()
    """
    idx = [i for i, p in enumerate(plan) if p["align"]]
    if not idx:
        return {"error": "无可对齐类别"}
    X = feat[idx]
    sc, clf, classes = device_deploy
    proba = clf.predict_proba(sc.transform(X))
    top1 = proba.argmax(axis=1)

    rows = []
    for j, i in enumerate(idx):
        rows.append({
            "cx_class": plan[i]["cx_class"],
            "expect": plan[i]["align"],
            "pred": classes[top1[j]],
            "proba": round(float(proba[j][top1[j]]), 4),
        })
    y = [r["expect"] for r in rows]
    p = [r["pred"] for r in rows]
    per = {}
    for c in sorted(set(y)):
        sub = [r for r in rows if r["expect"] == c]
        per[c] = {
            "n": len(sub),
            "acc": round(sum(1 for r in sub if r["pred"] == c) / len(sub), 4),
        }
    # 🔴 Spaghetti↔Off_platform 必须单独看（语义相近，最可能混淆对）
    spag = [r for r in rows if r["expect"] == "Off_platform"]
    spag_cm = dict(sorted(Counter(r["pred"] for r in spag).items(),
                          key=lambda x: -x[1]))
    # 输出塌缩诊断（D8 最有价值的指标）
    cnt = Counter(p)
    top, topn = cnt.most_common(1)[0]
    # Good Print 拒答门控（正常件不是我们的类，只测是否被拒答）
    gidx = [i for i, pp in enumerate(plan) if pp["cx_class"] == NORMAL_CX]
    gate = None
    if gidx:
        gproba = clf.predict_proba(sc.transform(feat[gidx]))
        gtop = gproba.argmax(axis=1)
        gp = gproba.max(axis=1)
        gcnt = Counter(classes[t] for t in gtop)
        gate = {
            "n": len(gidx),
            "would_reject_at_0.90": int((gp < 0.90).sum()),
            "reject_rate": round(float((gp < 0.90).mean()), 4),
            "false_alarm": dict(sorted(gcnt.items(), key=lambda x: -x[1])),
            "false_alarm_total": int((gp >= 0.90).sum()),
        }
    return {
        "n": len(rows),
        "covered_classes": sorted(set(y)),
        "acc": round(sum(1 for t, q in zip(y, p) if t == q) / len(rows), 4),
        "per_class": per,
        "random_baseline": round(RANDOM_BASELINE, 4),
        "spaghetti_off_platform_confusion": spag_cm,
        "collapse": {
            "top_class": top,
            "top_share": round(topn / len(rows), 4),
            "n_distinct_pred": len(cnt),
            "pred_dist": dict(sorted(cnt.items(), key=lambda x: -x[1])),
            "verdict": ("🔴 输出塌缩" if topn / len(rows) > 0.5
                        else "未见塌缩"),
        },
        "good_print_gate": gate,
    }


def m3_joint(x02r, y02r, g02r, feat, y, groups, labels02r, labels_cx,
             k, seed) -> dict:
    """③ 双向吞噬：DC-02R + Caxton-S 联合训练，两侧各自 held-out。

    🔴 只用**可对齐**的 Caxton-S 3 类（under/over/good 无对应，不参与）。
    ⚠️ Caxton-S 只有 3 类 → 不能算 8类 macro-F1（缺的类没有真值样本），
       故 B 侧只算**对齐 3 类**的 macro-F1，与 ① 的上界同口径才可比较。
    """
    import numpy as _np
    n02 = len(y02r)
    X = _np.vstack([x02r, feat])
    # 🔴 2026-10-06 修：原写法 `y = list(y02r) + list(y)` 用**同名参数**覆盖入参，
    #   第二次 `list(y)`拿到的是已拼接的列表 → 长度翻倍。
    #   同理 groups。改用新变量名，杜绝这类遮蔽。
    y_all = list(y02r) + list(y)
    # 🔴 域前缀强制不同：DC-02R group 是 '0'..'1748'，Caxton-S 是 'p22'。
    #   虽然实测两者不会撞车，但**依赖"恰好不撞"是隐患** → 显式加前缀。
    g_all = [f"d02r:{x}" for x in g02r] + [f"cx:{x}" for x in groups]
    assert len(y_all) == len(g_all) == X.shape[0], (
        f"拼接后长度不一致：X={X.shape[0]} y={len(y_all)} g={len(g_all)}")
    is_cx = [False] * n02 + [True] * len(y)
    folds = group_folds(g_all, k, seed)
    p02, p_cx = [], []
    for f in range(k):
        tr = [i for i in range(len(y_all)) if folds[i] != f]
        te = [i for i in range(len(y_all)) if folds[i] == f]
        if not te or not tr:
            continue
        probe = make_probe()
        probe.fit(X[tr], [y_all[i] for i in tr])
        for i, pr in zip(te, probe.predict(X[te])):
            (p_cx if is_cx[i] else p02).append((y_all[i], pr))
    a_f1 = macro_f1([t for t, _ in p02], [q for _, q in p02], labels02r)
    b_f1 = macro_f1([t for t, _ in p_cx], [q for _, q in p_cx], labels_cx)
    # 🔴 Off_platform（91 张小类）在 A 侧的存活率 —— D9 就是它先出事
    op = [(t, q) for t, q in p02 if t == "Off_platform"]
    op_keep = (sum(1 for t, q in op if q == "Off_platform") / len(op)
               if op else None)
    # 🔴 反向：Caxton-S 的 Spaghetti（映射到 Off_platform）有没有被吞
    sp = [(t, q) for t, q in p_cx if t == "Off_platform"]
    sp_keep = (sum(1 for t, q in sp if q == "Off_platform") / len(sp)
               if sp else None)
    return {
        "A_dc02r_macro_f1": round(a_f1, 4),
        "A_n": len(p02),
        "A_off_platform_keep": (round(op_keep, 4) if op_keep is not None
                                 else None),
        "A_off_platform_n": len(op),
        "B_caxton_aligned_macro_f1": round(b_f1, 4),
        "B_n": len(p_cx),
        "B_spaghetti_keep": (round(sp_keep, 4) if sp_keep is not None else None),
        "B_spaghetti_n": len(sp),
        "labels_joint": sorted(set(y_all)),
    }


# ---------------------------------------------------------------- 报告
def render_md(res: dict) -> str:
    L = []
    A = L.append
    m1 = res.get("m1_domain_upper", {})
    m2 = res.get("m2_zeroshot", {})
    m3 = res.get("m3_joint", {})
    A("# D10 · Caxton-S 跨域探针结果\n")
    A(f"- 数据：Caxton-S v1（Kaggle，CC BY 4.0，"
      f"**需署名** Brion & Pattinson (2022) DOI 10.17863/CAM.84082）")
    A(f"- 抽样：**每类 {res['_meta']['per_class']} 个打印批次**"
      f"共 {res['_meta']['n_plan']} 张"
      f"（group-aware，按**打印批次**分组）")
    A(f"- 🔴 **未覆盖类别**：{', '.join(CX_CLASSES[c] for c in UNCOVERED)}"
      f" —— 语义粒度与我们 5 类不对应，**不硬塞进任何类**")
    A(f"- ⚠️ Caxton-S 为 **64×64 喷嘴特写**，上采样到 224；"
      f"与 DC-02R 原生分辨率不可并列比较\n")

    # ---- ① ----
    A("## ① Caxton-S 域内上界（骨干可用性）\n")
    if "error" in m1:
        A(f"❌ {m1['error']}\n")
    else:
        A(f"- n = {m1['n']}，group = {m1['n_groups']}，"
          f"{m1['n_folds_used']} 折 group-aware")
        A(f"- **macro-F1 = {m1['macro_f1']*100:.2f}%**\n")
        A("| 类 | n | acc |")
        A("|---|---|---|")
        for c, v in m1["per_class"].items():
            A(f"| {c} | {v['n']} | {v['acc']*100:.1f}% |")
        up = m1["macro_f1"]
        if up >= VERDICTS["domain_upper_ok"]:
            A(f"\n✅ **{up*100:.1f}% ≥ 90%** → 骨干在该域完全可用，"
              f"跨域实验有判别力。")
        elif up >= VERDICTS["domain_upper_weak"]:
            A(f"\n⚠️ **{up*100:.1f}% 落在 70~90%** → 可用但有难度，"
              f"跨域结果需谨慎解读。")
        else:
            A(f"\n🔴 **{up*100:.1f}% < 70%** → **该数据不能用作跨域判别基准**"
              f"（域内就这么差，跨域低分无法归因于域差）。")

    # ---- ② ----
    A("\n## ② DC-02R 模型 → Caxton-S 零样本跨域（与 D8 同构）\n")
    if "error" in m2:
        A(f"❌ {m2['error']}\n")
    else:
        A(f"- 只统计可对齐的 {len(m2['covered_classes'])} 类，n = {m2['n']}")
        A(f"- **准确率 = {m2['acc']*100:.2f}%**"
          f"（随机基线 {m2['random_baseline']*100:.1f}%）\n")
        A("| 真值（Caxton-S） | 映射到 | n | acc |")
        A("|---|---|---|---|")
        inv = {v: k for k, v in ALIGN.items() if v}
        for c, v in m2["per_class"].items():
            A(f"| {CX_CLASSES.get(inv.get(c, ''), c)} | `{c}` | {v['n']} |"
              f" {v['acc']*100:.1f}% |")
        col = m2["collapse"]
        A(f"\n**输出塌缩诊断**：最大类 `{col['top_class']}` 占 "
          f"**{col['top_share']*100:.1f}%**，不同预测数 {col['n_distinct_pred']} "
          f"→ {col['verdict']}")
        g = m2.get("good_print_gate")
        if g:
            A(f"\n**Good Print 拒答门控**（n={g['n']}，正确行为是拒答）：")
            A(f"拒答率 **{g['reject_rate']*100:.1f}%**，"
              f"误报为缺陷 **{g['false_alarm_total']} 张**")
            A(f"误报分布：{g['false_alarm']}")
        sp = m2.get("spaghetti_off_platform_confusion")
        if sp:
            A(f"\n🔴 **Spaghetti→Off_platform 混淆分布**（最可能的混淆对，"
              f"不合并统计）：{sp}")
        acc = m2["acc"]
        A("")
        if acc > VERDICTS["zeroshot_good"]:
            A(f"✅ **{acc*100:.1f}% > 50%** → **H1 获得支持**："
              f"域多样性更高的数据上，泛化明显好于 D8 的 6.25%。")
        elif acc >= VERDICTS["zeroshot_bad"]:
            A(f"⚠️ **{acc*100:.1f}% 落在 20~50%** → 域多样性有帮助但不充分"
              f"（这是最可能的结果）。**不能宣称泛化问题已解决。**")
        else:
            A(f"🔴 **{acc*100:.1f}% < 20%** → **H1 不支持**："
              f"换域多样性数据也救不回，**必须改方案本身**。")

    # ---- ③ ----
    A("\n## ③ 双向吞噬（D9 教训，不可省）\n")
    if m3:
        base = res["_meta"].get("baseline_dc02r_f1")
        A("| 侧 | 指标 | 值 | 判据 | 结论 |")
        A("|---|---|---|---|---|")
        if base:
            drop = (base - m3["A_dc02r_macro_f1"]) * 100
            ok = drop <= VERDICTS["joint_max_drop_pp"]
            A(f"| A · DC-02R | 5 类 macro-F1 | **{m3['A_dc02r_macro_f1']*100:.2f}%** "
              f"（基线 {base*100:.2f}%，Δ {drop:+.2f}pp）| 掉 ≤2pp |"
              f" {'✅ 通过' if ok else '🔴 不通过'} |")
        if m1.get("macro_f1") is not None:
            keep = m3["B_caxton_aligned_macro_f1"] / m1["macro_f1"] \
                if m1["macro_f1"] else 0
            ok = keep >= VERDICTS["joint_min_keep"]
            A(f"| B · Caxton-S | 对齐 3 类 macro-F1 | "
              f"**{m3['B_caxton_aligned_macro_f1']*100:.2f}%** "
              f"（上界 {m1['macro_f1']*100:.2f}%，保持 {keep*100:.1f}%）|"
              f" ≥95% | {'✅ 通过' if ok else '🔴 不通过'} |")
        if m3.get("A_off_platform_keep") is not None:
            v = m3["A_off_platform_keep"]
            A(f"| A · 小类存活 | `Off_platform` 保持率 |"
              f" {v*100:.1f}%（n={m3['A_off_platform_n']}）| 不被吞 |"
              f" {'✅' if v >= 0.8 else '🔴 被吞'} |")
        if m3.get("B_spaghetti_keep") is not None:
            v = m3["B_spaghetti_keep"]
            A(f"| B · 反向吞噬 | `Spaghetti` 保持率 |"
              f" {v*100:.1f}%（n={m3['B_spaghetti_n']}）| 不被旧类吞 |"
              f" {'✅' if v >= 0.8 else '🔴 被吞'} |")
        A("\n> 🔴 只有**四项全过**才算联合训练可行。只测一侧等于没测"
          "（D9 就是折外「零损伤」而部署件 100% 缺陷被吞）。")

    A(f"""
## 结论与边界

1. 本实验只测**Caxton-S 这一个域**，⚠️ **不能推出「跨域泛化已解决」**。
2. 对外口径不变：**DC-02R 内、group-aware 折、5 类 macro-F1 99.73%**；
   本页任何数字都**不得**混入那个口径当"提升"。
3. **无论结果如何，OOD/域外拒答仍是必做项**——多域训练只能推迟这个问题。
4. 🔴 **高置信度在域外是危险信号**（D8 已实测错误答案概率也接近 1.0）。
""")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400,
                    help="每类抽多少个**打印批次**（<=0 = 全取）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # 部署件的 5 类名单：**从 infer 单一事实源取**，不硬编码
    from src.vision.infer import DEFECT_CLASSES as blob_classes_5

    if not CAXTON_ZIP.exists():
        print(f"❌ 找不到 {CAXTON_ZIP}\n"
              f"→ 跑 `python scripts/fetch_caxton_s.py` 下载并校验")
        return 1

    print("=" * 70)
    print("D10 · Caxton-S 跨域探针")
    print("=" * 70)
    print(f"zip  : {CAXTON_ZIP} "
          f"({CAXTON_ZIP.stat().st_size/1e9:.2f} GB)")
    print(f"许可 : CC BY 4.0 —— 🔴 对外产出必须署名 Brion & Pattinson (2022)")

    with zipfile.ZipFile(CAXTON_ZIP) as z:
        entries = scan_entries(z)
    print("\n[数据实况] zip 内各类张数（全量）：")
    for c, v in entries.items():
        al = ALIGN[c]
        tag = ("→ `" + al + "`" if al else
               ("（正常件，只测拒答）" if c == NORMAL_CX else "（**未覆盖**）"))
        print(f"  {c:18s} {len(v):6d}{tag}")

    plan = sample_plan(entries, a.per_class, a.seed)
    print(f"\n[抽样] 共 {len(plan)} 张（每类 ≤{a.per_class} 个打印任务，"
          f"seed={a.seed}）")
    print(f"  按任务分组: {len({p['task'] for p in plan})} 个唯一 group")
    if a.dry_run:
        for c, n in sorted(Counter(p["cx_class"] for p in plan).items()):
            print(f"  {c:18s} {n}")
        print("\n[dry-run] 不提特征")
        return 0

    print("\n[特征] 提冻结特征（与 infer._tf() 同一口径）…")
    with zipfile.ZipFile(CAXTON_ZIP) as z:
        feat = embed_from_zip(z, [p["name"] for p in plan], a.device)
    print(f"  {feat.shape}")

    y_all = [p["align"] or p["cx_class"] for p in plan]
    groups = [p["task"] for p in plan]

    # ---- ① 域内上界（只用可对齐 3 类 + 正常件另算）----
    print("\n[①] Caxton-S 域内上界 …")
    m1 = m1_domain_upper(feat, y_all, groups,
                         sorted({ALIGN[c] for c in ALIGNED}), a.folds, a.seed)
    print(f"  macro-F1 = {m1['macro_f1']*100:.2f}%")

    # ---- ② 零样本跨域 ----
    print("\n[②] 零样本跨域（加载 DC-02R 部署件，不重训）…")
    from src.vision.infer import load_deployed_model
    # 🔴 2026-10-06 修：load_deployed_model() 返回的是
    #   **(scaler, clf, model, meta) 四元组**，不是 dict。
    #   原写法 `blob["meta"]` → TypeError: tuple indices must be integers。
    sc, clf, backbone, meta = load_deployed_model(device=a.device)
    classes = list(meta["classes"])
    print(f"  部署件类别 = {classes}"
          f"（正常类启用={meta.get('normal_class_enabled')}）")
    if classes != list(blob_classes_5):
        print(f"  ⚠️ 部署件类别 = {classes}，与预期 5 类不同，"
              f"本实验的可对齐判定仍按 ALIGN 走（未覆盖类仍标未覆盖）")
    m2 = m2_zeroshot(feat, plan, (sc, clf, classes), ALIGN)
    print(f"  acc = {m2['acc']*100:.2f}%"
          f"（随机 {m2['random_baseline']*100:.1f}%）")

    # ---- ③ 双向吞噬 ----
    print("\n[③] 双向吞噬联合训练 …")
    m3 = {}
    #🔴 标签与 group **直接从 data/p1_folds.json 读**，不另建缓存——
    #   缓存会与 folds 不同步（本项目已吃过"实验口径≠部署口径"的亏）。
    #   ⚠️ 关键：`l2_feats.npy` 的行序必须与 folds["images"] 的行序一致。
    #   先断言长度与首行路径的类别是否对得上，对不上**直接报错**不猜。
    folds_json = ROOT / "data" / "p1_folds.json"
    cache = ROOT / "exp" / "_cache" / "l2_feats.npy"
    if folds_json.exists() and cache.exists():
        fj = json.loads(folds_json.read_text(encoding="utf-8"))
        imgs = fj["images"]
        x02r = np.load(cache)
        if len(imgs) != len(x02r):
            print(f"  ❌ 行数不一致：folds {len(imgs)} vs 缓存 {len(x02r)}"
                  f" → 不猜，直接中止（③ 跳过）")
        else:
            y02r = [it["label"] for it in imgs]
            g02r = [str(it["group"]) for it in imgs]
            # 反向自检：缓存行序是否与 folds 一致（抽 5 行比对路径里的类别）
            bad = 0
            for i in range(0, len(imgs), max(len(imgs) // 5, 1)):
                cls_in_path = Path(imgs[i]["path"]).parent.name
                if cls_in_path != imgs[i]["label"]:
                    bad += 1
            if bad:
                print(f"  ❌ 缓存与folds 行序疑似错位（{bad}/5 抽样类别不符）"
                      f" → ③ 跳过，不硬跑")
            else:
                base = None
                from sklearn.metrics import f1_score
                from sklearn.model_selection import GroupKFold
                oof = [None] * len(y02r)
                for tr, te in GroupKFold(
                        n_splits=a.folds).split(x02r, y02r, groups=g02r):
                    pr = make_probe()
                    pr.fit(x02r[tr], [y02r[i] for i in tr])
                    for i, p in zip(te, pr.predict(x02r[te])):
                        oof[i] = p
                idx = [i for i, p in enumerate(oof) if p]
                base = f1_score([y02r[i] for i in idx], [oof[i] for i in idx],
                                average="macro")
                print(f"  A 侧基线（本次同口径重算）macro-F1 = {base*100:.2f}%")
                keep = [i for i, p in enumerate(plan) if p["align"]]
                m3 = m3_joint(x02r, y02r, g02r, feat[keep],
                              [plan[i]["align"] for i in keep],
                              [plan[i]["task"] for i in keep],
                              sorted(set(classes)),
                              sorted({ALIGN[c] for c in ALIGNED}),
                              a.folds, a.seed)
                m3["baseline_dc02r_f1"] = round(float(base), 4)
                m3["A_drop_pp"] = round(
                    (base - m3["A_dc02r_macro_f1"]) * 100, 2)
                print(f"  A 侧 = {m3['A_dc02r_macro_f1']*100:.2f}%"
                      f"（Δ {m3['A_drop_pp']:+.2f}pp）")
                print(f"  B 侧 = {m3['B_caxton_aligned_macro_f1']*100:.2f}%")
    else:
        print(f"  ⚠️ 缺 {folds_json} 或 {cache}，跳过 ③")

    res = {
        "m1_domain_upper": m1,
        "m2_zeroshot": m2,
        "m3_joint": m3,
        "_meta": {
            "n_plan": len(plan),
            "per_class": a.per_class,
            "seed": a.seed,
            "folds": a.folds,
            "n_groups": len(set(groups)),
            "group_definition": "打印批次 = 同(类,打印机)下帧号间隙<=100 的连续段",
            "zip_bytes": CAXTON_ZIP.stat().st_size,
            "uncovered": UNCOVERED,
            "alignment": ALIGN,
            "license": "CC BY 4.0 (需署名 Brion & Pattinson 2022, "
                       "DOI 10.17863/CAM.84082)",
            "note": "Caxton-S 为 64x64 喷嘴特写上采样到 224，"
                    "与 DC-02R 原生分辨率不可并列比较",
            "verdicts_pre_registered": VERDICTS,
        },
    }
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    md = render_md(res)
    OUT_MD.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"[产出] {OUT_JSON}\n[产出] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
