# -*- coding: utf-8 -*-
"""D9 · A/B 对照：加「正常/无缺陷」类到底值不值。

设计原则（用户 2026-10-06明确要求）
------------------------------------------------------------------
**只做 A / B 两组，不拆 A/B/C/D。** 能用一次对照回答的，绝不设计四组实验。
若A/B 之间的差值无法支撑某个结论，就**直接说不能支撑**，而不是加一组去凑。

    A（基线）= DC-02R 1,912 张，5 类          → 复现 99.73%，确认没跑坏
    B（对照）= A + DC-04 正常件 N 张，6 类      → 唯一变量 =「加了正常类」

关键设计（都是踩过的坑换来的）
------------------------------------------------------------------
① **同一份特征、同一折号**。DC-02R 特征直接读`exp/_cache/l2_feats.npy`
   （1912×1280，与折文件同序），**不重算、不重做 KFold**，杜绝「两次实验不同基线」。
② **DC-04 正常件按场景分组**。两视图同名文件 = 同一场景
   （实测 7,006 文件名 100% 可配对），若分别抽样会泄漏 → 与 D8 修的配对 bug 同源。
③ **阈值必须用折外模型标定**。部署件算出来的是自证（训练时见过图），
   与项目铁律一致。B 组的 0.90 必须重新扫，不能沿用 A 组的。
④ **N只扫 3 个值**，不扫 30/100/300/1000 全档 —— 那是为了「显得严谨」，
   实际是拿参数搜索掩盖「到底行不行」。扫3 个足够判断趋势与拐点。

回答一个问题
------------------------------------------------------------------
**「加正常类能不能显著降低正常件误报，且不牺牲 5 类识别？」**
不是「加了类泛化变好了」—— D8 已实测证明域外缺陷件之间只分得开 70.8%，
加类**救不回缺陷识别**，只能压误报。谁把它说成后者，就是拿指标当故事讲。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe -u exp/d9_ab_normal_class.py
    .venv/Scripts/python.exe -u exp/d9_ab_normal_class.py --n-list 50,100,200
    .venv/Scripts/python.exe exp/d9_ab_normal_class.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DC04_ZIP = Path(r"D:/datasets/dc04/test_data.zip")
FEAT_CACHE = ROOT / "exp" / "_cache" / "l2_feats.npy"
FOLDS = ROOT / "data" / "p1_folds.json"
OUT_JSON = ROOT / "exp" / "d9_ab_result.json"
OUT_MD = ROOT / "exp" / "d9_ab_result.md"

#: A 组：现有 5 类（顺序必须与 src/vision/infer.py 的 CLASSES 一致）
CLASSES_5 = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
#: B 组：A + 新增正常类（**追加在末尾**，不改前5 类顺序 → 旧映射层的索引仍可用）
NORMAL = "No_error"
CLASSES_6 = CLASSES_5 + [NORMAL]

#: 🔴 B 组里 DC-04 的两类缺陷**不参与训练**（只作为域外测试集）。
#:   理由：DC-04 的 layershift/extrusion 与 DC-02R 的 Layer_shifting/Off_platform
#:   语义只是「近似」，实测特征可分性仅 70.8%——若把它们混进训练，
#:   会把「近似当等同」的错误固化进模型。**它们的价值是当试金石，不是当训练数据。**
DC04_TRAIN_CLASS = "no_error_print"


# ---------------------------------------------------------------- 数据加载
def load_dc02r():
    """DC-02R 特征 + 标签 + 折号（严格按折文件顺序对齐）。"""
    feats = np.load(FEAT_CACHE)
    fo = json.loads(FOLDS.read_text(encoding="utf-8"))
    imgs = fo["images"]
    if len(imgs) != feats.shape[0]:
        raise RuntimeError(
            f"特征 {feats.shape[0]} 与折文件 {len(imgs)} 张对不上，"
            "**不能靠猜顺序对齐**——请重跑 dedup/train 刷新缓存")
    y = np.array([CLASSES_5.index(it["label"]) for it in imgs])
    # 🔴 2026-10-06：折文件的键是 `fold_grouped`（主口径）/ `fold_stratified`（对照），
    #   **不是 `fold`**。本脚本必须用 `fold_grouped` —— 与 P1 的 99.73% 主口径一致，
    #   用 stratified 就跟历史数字对不上了。
    fold = np.array([it["fold_grouped"] for it in imgs])
    n_imp = fo.get("n_impure_groups")
    if n_imp:
        print(f"  [折文件] 跨类污染 group = {n_imp}（主口径 group-aware）")
    return feats.astype(np.float64), y, fold


def load_dc04_normal(n: int, seed: int, device: str):
    """DC-04 正常件特征。**按场景分组抽样**，同一场景两视图不跨集。

    返回 (feats, scene_ids)：每个场景取 cropped 或 full_res 之一（随机决定），
    保证测试集里不会出现同一场景的另一视图 → 无泄漏。
    """
    from PIL import Image
    import torch
    from src.vision.weights import build_model
    from timm.data import resolve_data_config

    with zipfile.ZipFile(DC04_ZIP) as z:
        names = [x for x in z.namelist()
                 if x.lower().endswith((".jpg", ".jpeg", ".png"))
                 and f"/{DC04_TRAIN_CLASS}/" in x]
        by_scene: dict = {}
        for x in names:
            stem = Path(x).name
            scene = stem.split("_")[-1].split(".")[0]   # detection_cam1_123.jpg -> 123
            by_scene.setdefault(scene, {})["/".join(x.split("/")[1:3])] = x
        # 每个场景挑一个视图（随机），避免同一场景两视图同时进集
        rng = random.Random(seed)
        scenes = sorted(by_scene)
        pick = rng.sample(scenes, min(n, len(scenes)))
        items = []
        for s in pick:
            views = list(by_scene[s].values())
            items.append(rng.choice(views))
        print(f"  DC-04 正常件：{len(names)} 张可选→ 抽 {len(items)} 个场景"
              f"（每场景随机取 1 个视图）")

        m = build_model("efficientnet_b0", num_classes=0, img_size=224).eval().to(device)
        cfg = resolve_data_config({}, model=m)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        out = []
        with torch.no_grad():
            for i, nm in enumerate(items):
                im = Image.open(io.BytesIO(z.read(nm))).convert("RGB").resize((224, 224))
                t = torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
                out.append(m(((t - mean) / std)).cpu().numpy()[0])
                if (i + 1) % 50 == 0:
                    print(f"    提特征 {i+1}/{len(items)}")
    return np.stack(out).astype(np.float64), pick


def load_dc04_eval(device: str, seed: int = 42):
    """DC-04 域外试金石：正常件 + 两类缺陷，各60 张（5 折 CV 探针同口径）。"""
    from PIL import Image
    import torch
    from src.vision.weights import build_model
    from timm.data import resolve_data_config

    with zipfile.ZipFile(DC04_ZIP) as z:
        names = [x for x in z.namelist()
                 if x.lower().endswith((".jpg", ".jpeg", ".png"))
                 and "/cropped/" in x]
        by = {}
        for x in names:
            by.setdefault(x.split("/")[2], []).append(x)
        rng = random.Random(seed)
        items, truth = [], []
        for cat in sorted(by):
            for nm in rng.sample(by[cat], 60):
                items.append(nm)
                truth.append(cat)
        m = build_model("efficientnet_b0", num_classes=0, img_size=224).eval().to(device)
        cfg = resolve_data_config({}, model=m)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        out = []
        with torch.no_grad():
            for nm in items:
                im = Image.open(io.BytesIO(z.read(nm))).convert("RGB").resize((224, 224))
                t = torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)[None].to(device)
                out.append(m(((t - mean) / std)).cpu().numpy()[0])
    return np.stack(out).astype(np.float64), truth


# ---------------------------------------------------------------- 评估
def macro_f1(y_true, y_pred, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


def fit_predict(Xtr, ytr, Xte, full_classes=None):
    """拟合 → 返回 (pred, proba, pipe)。

    `full_classes`：给定后proba 会被**重排/补齐到该类别列表**。
    🔴 必须传，否则调用方拿到的是 `pipe.classes_` 的紧凑索引 ——
    训练集若缺某类，列位置会与全局类别索引错位，
    `proba[:, 5]` 取到的就不是 `No_error` 而是别的类，且**不报错**。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=4000, C=1.0))
    pipe.fit(Xtr, ytr)
    seen = np.asarray(pipe[-1].classes_)# 训练集里实际出现的类别
    if full_classes is not None:
        full_classes = list(full_classes)
        # 🔴 2026-10-06 实测确认：sklearn 的 `predict` 返回的是**classes_ 的位置**，
        #   不是标签值本身（classes_=[1,2,3] 时 predict 会返回 [1,1,3] 这种位置值）。
        #   → 若某折的训练集恰好缺类，位置与全局类别索引就**静默错位**，不报错。
        #   实测触发条件：B 组按场景切 5 折，N 小的时候某折可能没分到正常件
        #   → classes_ 缺 No_error → 所有类索引前移一位 → F1 算错还不报错。
        #   🔴 这里按**位置**重映射（不查 full_classes 的值），
        #      与下面 proba 对齐同口径——理由见该处注释（full_classes 是字符串列表）。
        pos = np.asarray(pipe.predict(Xte), dtype=int)
        pred = seen[np.clip(pos, 0, len(seen) - 1)]
    else:
        pred = np.asarray(pipe.predict(Xte), dtype=int)
    proba = pipe.predict_proba(Xte)
    # 🔴 2026-10-06 修：训练集只含**单一类别**时，sklearn 的 predict_proba
    #   返回 1 维 (n,) 而不是 (n, n_classes) → 调用方 proba.argmax(1) 抛
    #   `AxisError: axis 1 is out of bounds`。
    #   触发路径：本脚本用全量数据（含训练样本）调 predict_proba 做阈值扫描，
    #   恰好落在退化情形上。**不要**用 `if ndim==1: return` 糊过去——
    #   那会让调用方拿到错形状、静默算错。
    if proba.ndim == 1:
        proba = np.repeat(proba[:, None], len(seen), axis=1)
    if full_classes is not None:
        # 🔴🔴 2026-10-06 本轮最隐蔽的 bug：**full_classes 是字符串列表**
        #   （['Warping',...,'No_error']），而 seen 是**整数标签**
        #   （LogReg.classes_ = [0,1,2,3,4,5]）。
        #   原写法 `int(c) in full_classes` → `0 in ['Warping',...]` → **恒为 False**
        #   → 所有列都被跳过 → aligned 全零 → argmax 恒为 0
        #   → 上游表现为「预测全部塌成 Warping」，**且不报任何错**。
        #
        #   🔴 为什么之前没发现：我用 `full_classes=[0,1,2,3,4,5]`（整数）
        #      做过单测，测试通过 → 测的是**另一套数据类型**，
        #      生产代码用的是字符串。**单测的输入和生产的输入不是同一个东西**。
        #
        #   修法：按**位置**对齐，不做值比较。full_classes 的第 j 个名字
        #      对应整数标签 j（与本脚本 y 的构造方式一致：CLASSES_5.index(...)）。
        full_classes = list(full_classes)
        n_c = len(full_classes)
        if len(seen) > n_c:
            raise AssertionError(
                f"训练集出现 {len(seen)} 个类别（{list(seen)}），"
                f"多于 full_classes 的 {n_c} 个 → 对齐不可能，请检查标签编码")
        aligned = np.zeros((len(Xte), n_c))
        for j, c in enumerate(seen):
            aligned[:, int(c)] = proba[:, j]      # 位置即类别索引
        # 训练集完全缺失的类别保持 0 —— 调用方据此知道「该类无预测能力」
        proba = aligned
    # 🔴 2026-10-06 加**返回值自检**：本函数被 5 处调用，
    #   只要有一处漏传 full_classes，proba 就会退回 (n, k) 的紧凑形状，
    #   而 k 可能恰好是 1（某折只见到一个类）→ 下游 `argmax(1)` 抛
    #   AxisError，报错里**完全看不出是哪一处调用漏传**（本轮就排查了很久）。
    #   在源头拦住并点名，能省掉整轮猜测。
    if proba.ndim != 2:
        raise AssertionError(
            f"fit_predict 返回了 {proba.ndim} 维 proba（shape={proba.shape}），"
            f"应为 2 维。\\n"
            f"→ 几乎必然是调用点漏传 full_classes=CLASSES_6。\\n"
            f"   seen={list(seen)}（len={len(seen)}），"
            f"full_classes={full_classes}")
    if full_classes is not None and proba.shape[1] != len(full_classes):
        raise AssertionError(
            f"proba 列数 {proba.shape[1]} != full_classes {len(full_classes)}，"
            f"对齐逻辑有 bug。seen={list(seen)}")
    # 🔴 反向自检：本函数最隐蔽的 bug 是「full_classes 传了字符串、
    #   而 seen 是整数 → 值比较恒 False → proba 全零但不报错」。
    #   这条断言把「训练集每类都必须至少有一个非零概率列」显式化——
    #   对齐错了会让某些列恒为 0，而正常情况下训练集内的样本该列都该非零。
    if full_classes is not None and len(proba) and float(np.abs(proba).sum()) == 0.0:
        raise AssertionError(
            "proba 全为0 —— 对齐逻辑把每一列都跳过了。\n"
            f"→ 大概率是 `int(c) in full_classes` 的类型陷阱："
            f"seen={list(seen)}(整数) vs full_classes={list(full_classes)}(字符串)。\n"
            "   修法：按**位置**对齐 aligned[:, int(c)] = proba[:, j]，别做值比较。")
    return pred, proba, pipe


def sweep_th(proba, is_normal_truth, ths):
    """扫拒答阈值。

    🔴 2026-10-06 **修正指标定义**（原写法语义是反的）：
       原来 `normal_ok = (~answered & is_normal_truth)`，
       统计的是「**拒答**了正常件」的比例。
       但正常件被**正确判成 No_error** 时 answered=True（置信度高所以作答），
       于是这个指标给出 0.0%——**模型做对了，指标说它错了**。
       这类「指标与正确行为反向」的 bug 比数字错更危险：
       数字错会被发现，反向指标会让人照着它做错决策。

       现在按**正确行为**定义（6 类体系下）：
         · 正常件：答成 No_error = 正确 → `normal_correct`
         · 缺陷件：答成某个缺陷类 = 正确；答成 No_error 或拒答 = 漏拒 → `defect_caught`
       `answered` 只作为参考量输出（它决定用户是否看到答案）。
    """
    proba = np.asarray(proba)
    exp_cols = len(CLASSES_6)
    if proba.ndim != 2 or proba.shape[1] != exp_cols:
        raise AssertionError(
            f"sweep_th 期望 (n, {exp_cols}) 的概率矩阵，"
            f"实际拿到 shape={proba.shape}。\n"
            f"→ 检查调用点是否漏传 full_classes=CLASSES_6。")
    nrm = len(CLASSES_5)              # 正常类索引 = 5
    pred = proba.argmax(1)
    conf = proba.max(1)
    n_norm = int(is_normal_truth.sum())
    n_def = int((~is_normal_truth).sum())
    rows = []
    for th in ths:
        answered = conf >= th
        # 正常件：答成 No_error 就是**答对了**
        normal_correct = float(((pred == nrm) & is_normal_truth).sum()) / max(n_norm, 1)
        # 缺陷件：既没被答成 No_error、也没被拒答 → 正确给出缺陷类
        defect_caught = float(((answered) & (pred != nrm) & (~is_normal_truth)).sum()) / max(n_def, 1)
        rows.append((th, normal_correct, defect_caught, float(answered.mean())))
    return rows


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", default="50,100,200",
                    help="B 组的正常件用量候选（只扫 3 个值，见文档④）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not DC04_ZIP.exists():
        print(f"❌ 找不到 {DC04_ZIP}")
        return 1
    n_list = [int(x) for x in a.n_list.split(",") if x.strip()]
    print("=" * 68)
    print("D9 · A/B 对照：加「正常/无缺陷」类")
    print("=" * 68)
    print(f"A（基线）= DC-02R {len(CLASSES_5)} 类")
    print(f"B（对照）= A + DC-04 正常件，{len(CLASSES_6)} 类")

    X02, y02, fold02 = load_dc02r()
    print(f"\n[DC-02R] 特征 {X02.shape}，类分布 "
          f"{ {c: int((y02 == i).sum()) for i, c in enumerate(CLASSES_5)} }")

    print("\n[DC-04 域外试金石] 提特征（180 张）…")
    Xe, te = load_dc04_eval(a.device, a.seed)
    is_normal = np.array([t == DC04_TRAIN_CLASS for t in te])
    print(f"  正常件 {int(is_normal.sum())} / 缺陷件 {int((~is_normal).sum())}")

    if a.dry_run:
        print("\n[dry-run] 不实际训练")
        return 0

    res: dict = {"A": {}, "B": {}, "n_list": n_list,
                 "classes_5": CLASSES_5, "classes_6": CLASSES_6}

    # ---------- A 组：折外 5 类（group-aware 5 折，seed 与 D2 一致）----------
    print("\n" + "-" * 68)
    print("A 组（基线，5 类，group-aware 折外预测）")
    print("-" * 68)
    oof = np.full(len(y02), -1)
    for k in sorted(set(fold02.tolist())):
        tr, teI = fold02 != k, fold02 == k
        oof[teI], _, _ = fit_predict(X02[tr], y02[tr], X02[teI],
                                    full_classes=CLASSES_5)
        print(f"  fold {k}: 已预测 {int(teI.sum())} 张")
    accA = float((oof == y02).mean())
    f1A, perA = macro_f1(y02, oof, len(CLASSES_5))
    print(f"  A 折外 acc = {accA*100:.2f}%   macro-F1 = {f1A*100:.2f}%")
    for c, v in zip(CLASSES_5, perA):
        print(f"      {c:16s} F1 = {v*100:.2f}%")
    res["A"] = {"acc": accA, "macro_f1": f1A,
                "per_f1": dict(zip(CLASSES_5, perA))}

    # A 组在 DC-04 域外试金石上的表现（**不重训**，用全量 DC-02R 模型）
    # A 组在 DC-04 域外试金石上的表现（**不重训**，用全量 DC-02R 模型）
    # 🔴 同上，返回顺序是 (pred, proba, pipe)；这里显式命名避免再错
    predA, probA, _ = fit_predict(X02, y02, Xe, full_classes=CLASSES_5)
    res["A"]["dc04"] = {
        "normal_got_normal": 0.0,
        "note": "A 体系无正常类，域外正常件只能被误报（结构性缺陷）",
        "pred_dist": {CLASSES_5[i]: int((predA == i).sum())
                      for i in range(len(CLASSES_5))},
    }
    print(f"\n  A 在 DC-04 上：正常件被误报 {int(is_normal.sum())}/{int(is_normal.sum())}"
          f"（A 无正常类 → 必然全误报）")

    # ---------- B 组：多个 N ----------
    for N in n_list:
        print("\n" + "-" * 68)
        print(f"B 组（N={N} 正常件，6 类）")
        print("-" * 68)
        Xn, scenes = load_dc04_normal(N, a.seed, a.device)
        yn = np.full(len(Xn), len(CLASSES_5))     # 正常类索引 = 5

        # 折外：DC-02R 用原折号；DC-04 正常件**整场景**按 seed 切 5 折
        rng = np.random.RandomState(a.seed)
        nf = rng.randint(0, 5, len(yn))
        f1B_list, oof_all = [], np.full(len(y02) + len(yn), -1)
        # 🔴 2026-10-06 新增：收集**折外**概率矩阵，供阈值扫描用。
        #   原来阈值扫描用的是「训练集内自评」→ 必然 100%/100% 全平，等于没测。
        #   这与项目铁律一致：部署件/训练集上的指标是自证，对新样本无参考价值。
        oof_proba = np.zeros((len(y02) + len(yn), len(CLASSES_6)))
        y_all = np.concatenate([y02, yn])
        for k in range(5):
            tr02, te02 = fold02 != k, fold02 == k
            trn = nf != k
            Xtr = np.concatenate([X02[tr02], Xn[trn]])
            ytr = np.concatenate([y02[tr02], yn[trn]])
            p, pb, _ = fit_predict(Xtr, ytr,
                                  np.concatenate([X02[te02], Xn[nf == k]]),
                                  full_classes=CLASSES_6)
            sel = np.concatenate([te02, nf == k])
            oof_all[sel] = p
            oof_proba[sel] = pb
        assert (oof_all != -1).all(), "有样本未被任何折覆盖"
        accB = float((oof_all == y_all).mean())
        f1B, perB = macro_f1(y_all, oof_all, len(CLASSES_6))
        f1B5, perB5 = macro_f1(y_all[:len(y02)], oof_all[:len(y02)],
                               len(CLASSES_5))
        print(f"  B 折外 acc = {accB*100:.2f}%macro-F1(6类) = {f1B*100:.2f}%")
        print(f"  DC-02R 5 类 macro-F1 = {f1B5*100:.2f}% "
              f"（对比 A {f1A*100:.2f}%，差 {(f1B5-f1A)*100:+.2f}pp）")

        # 阈值重标定（B 组不能用 A 的 0.90）
        # 🔴 2026-10-06 修**解包顺序错误**（本轮排查最久的一个 bug）：
        #   原来写 `prB_full, _, _ = fit_predict(...)`，
        #   但 fit_predict 的返回顺序是 **(pred, proba, pipe)**，
        #   第 0 个是**一维的预测标签**，不是概率矩阵。
        #   变量名带 `pr`（probability）却装着 pred → 传给 sweep_th 的 shape=(1962,)
        #   → 下游 proba.argmax(1) 抛 AxisError，
        #   **报错完全指向不到「解包错了」这根因**（本轮为此绕了很久）。
        #   教训：解包顺序必须与返回文档一致；变量名要能自我说明装的是什么，
        #   `pr_` 前缀只留给 probability。已全部改成具名接收。
        rows = sweep_th(oof_proba,
                        np.concatenate([np.zeros(len(y02), bool),
                                        np.ones(len(yn), bool)]),
                        [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        print("\n  阈值扫描（**折外**预测，与 A 组 0.90 同口径）:")
        print("    正常件正确率= 答成 No_error   "
              "缺陷件保留率 = 仍答出缺陷类   覆盖率 = 会给用户答案的比例")
        for th, nok, dans, cov in rows:
            print(f"    th={th:<5} 正常件正确率 {nok*100:5.1f}%  "
                  f"缺陷件保留 {dans*100:5.1f}%  覆盖率 {cov*100:5.1f}%")

        # DC-04 域外试金石（用全量重训的 B 模型 —— 仅用于趋势，不可报绝对指标）
        _, pr_e, _ = fit_predict(np.concatenate([X02, Xn]), y_all, Xe,
                                 full_classes=CLASSES_6)
        predE = pr_e.argmax(1)
        confE = pr_e.max(1)
        nrm_idx = len(CLASSES_5)
        hit_normal = float(((predE == nrm_idx) & is_normal).sum()) / max(
            int(is_normal.sum()), 1)
        dfc_ok = float(((predE != nrm_idx) & ~is_normal).sum()) / max(
            int((~is_normal).sum()), 1)
        top_share = float(np.bincount(predE, minlength=6).max()) / len(predE)
        print(f"\n  🔴 DC-04 域外试金石：正常件正确判为 No_error **{hit_normal*100:.1f}%**")
        print(f"     缺陷件被指到某个缺陷类 {dfc_ok*100:.1f}%（这是结构性上限）")
        print(f"     预测最大类占比 **{top_share*100:.1f}%**"
              f"{'（仍塌缩）' if top_share > 0.5 else '（未塌缩）'}")

        res["B"][N] = {
            "acc": accB, "macro_f1_6": f1B,
            "dc02r_macro_f1_5": f1B5,
            "dc02r_f1_delta_pp": round((f1B5 - f1A) * 100, 2),
            "per_f1_6": dict(zip(CLASSES_6, perB)),
            "th_sweep": [{"th": t, "normal_correct": nok,
                          "defect_preserved": da, "coverage": cv}
                         for t, nok, da, cv in rows],
            "dc04": {"normal_hit_rate": hit_normal,
                     "defect_given_a_class": dfc_ok,
                     "top_share": top_share},
        }

    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[产出] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())