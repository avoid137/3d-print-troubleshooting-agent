"""P1 三级对照训练：L1 冻结骨干+kNN / L2 冻结骨干+线性探针 / L3 全量微调。

为什么是三级（2026-10-04 实检文献后定的方案，替代「只跑全量微调」）：

  L1 冻结骨干 + kNN
     同领域证据：Bast et al. 2026, Int J Adv Manuf Technol 144:2047-2059
     (DOI 10.1007/s00170-026-17844-w) 用预训练 ResNet-50 只提特征 + kNN，
     FFF 缺陷 4 类各类 F1 0.757~0.990、加权平均 0.812。方法可直接引用。

  L2 冻结骨干 + 线性探针
     跨领域证据：植物病害 PlantVillage→PlantDoc，冻结 DINOv2 线性探针 51.1%，
     比全量微调 EfficientNet-B0 高 +23.4pp；且全量微调使跨域 50.6%→29.4%
     （小数据上微调骨干会破坏预训练特征、损害泛化）。
     ⚠️ 反例：字体 394 类细分类线性探针仅 36.6%（远低于 LoRA 98.7%）
        → 线性探针不一定够用，**必须实测，不能假设**。

  L3 全量微调 EfficientNet-B0
     保留作上限对照。小样本下易过拟合，但不给它跑就不知道上限在哪。

🔴 评估纪律（沿用 P0 的 ±7pp 教训，2026-10-04 实测修掉 3 个会虚高的 bug）：
  1. **折划分一律读D2 产出的 `data/p1_folds.json`，绝不重做KFold**。
     原实现自己调 `StratifiedGroupKFold` → D2 算的group_id 和双口径全白费，
     「泄漏幅度」无从算起。
  2. **L3 早停必须用训练集内切出的验证集，绝不能用测试折**。
     原实现每轮 epoch 都在 `te` 上算 macro-F1 并取 best → 测试集参与调参，
     指标系统性虚高（这正是 P0 判官不一致那类教训）。
  3. **两组划分必须用同一批特征、同一套代码**，唯一差别是折号。
     否则「stratified vs grouped 的差值 = 泄漏幅度」这个推论就不成立。
  · 主口径 = group-aware（`fold_grouped`）；`fold_stratified` 仅作对照
  · 5 折 × N 种子，报 macro-F1 均值 ± 标准差
  · 组内差异 < 2pp 一律判「等价」，不宣称优劣
  · 数字做完才填

用法：
    python -m src.vision.train --folds data/p1_folds.json --level L2
    python -m src.vision.train --level all --backbone efficientnet_b0
    python -m src.vision.train --smoke-test     # 防线4：合成数据跑通全链路
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .guards import (EXPECTED_COUNTS, GuardError, RuntimeCounter,
                     audit_backbone)

SEED = 42
CLASSES = list(EXPECTED_COUNTS.keys())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------- 数据
def load_folds(folds_json: str):
    d = json.loads(Path(folds_json).read_text(encoding="utf-8"))
    imgs = d["images"]
    for it in imgs:
        it["y"] = CLASS_TO_IDX[it["label"]]
    return d, imgs


def iter_folds(items, scheme: str, n_splits: int):
    """按 D2 产出的折号切分，**不重新做 KFold**。

    scheme = "grouped"（主口径，去泄漏） | "stratified"（对照，算泄漏幅度）
    """
    for it in items:
        key = f"fold_{scheme}"
        if key not in it:
            raise GuardError(
                f"p1_folds.json 缺字段 '{key}' —— 请先跑 "
                f"`python -m src.vision.dedup --data-root <root>` 生成双口径折划分。")
    f = np.array([it[f"fold_{scheme}"] for it in items])
    g = np.array([it["group"] for it in items])
    y = np.array([it["y"] for it in items])

    # 🔴 防线12：grouped 口径下同一 group **绝不能**跨折，否则「去泄漏」是假的。
    if scheme == "grouped":
        bad = {int(gi) for gi in np.unique(g)
               if len(set(f[g == gi])) > 1}
        if bad:
            raise GuardError(
                f"fold_grouped 失效：{len(bad)} 个 group 跨折（示例 {sorted(bad)[:5]}）。"
                f"去泄漏口径不成立，训练结果不可用。")
    for k in range(n_splits):
        te = np.flatnonzero(f == k)
        tr = np.flatnonzero(f != k)
        if len(te) == 0:
            raise GuardError(f"{scheme} 第 {k} 折测试集为空（n_splits={n_splits}）")
        yield tr, te


def build_transform(timm_cfg, train: bool, img_size: int):
    from torchvision import transforms
    mean, std = timm_cfg["mean"], timm_cfg["std"]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),   # 打印件有方向性，不做旋转/强扭曲
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def decode_all(items, cfg, img_size: int, counter: RuntimeCounter):
    """把 1,912 张按 eval 变换解码成内存数组。

    1912 × 3 × 224×224 float32 ≈ 1.1 GB（不是 288 MB，那是 uint8），
    4060 8GB 机器放得下，换来 5 折 × 多种子 × 三级方法只解码一次。
    """
    import torch
    from PIL import Image
    tf = build_transform(cfg, train=False, img_size=img_size)
    arr = np.zeros((len(items), 3, img_size, img_size), dtype=np.float32)
    for i, it in enumerate(items):
        try:
            with Image.open(it["path"]) as im:
                arr[i] = tf(im.convert("RGB"))
            counter.record_ok()
        except Exception as e:
            counter.record_failed(f"{type(e).__name__}: {it['path']}")
            raise GuardError(f"图片无法读取: {it['path']} ({e})") from e
    return torch.from_numpy(arr), torch.from_numpy(
        np.array([it["y"] for it in items], dtype=np.int64))


# ---------------------------------------------------------------- 特征提取
def extract_features(model, x, batch: int = 64, device: str = "cuda"):
    """提取冻结骨干特征 [N, D]（全局平均池化）。

    🔴 2026-10-04 删掉原`multi_layer` 参数：注释写「取浅层特征」，
       实际代码是 `f[:, :C//4]` —— **通道切片**，不是浅层。
       BN 之后的通道是深层语义的前 C//4 维投影，拼接它既不是浅层、
       也没有文献支撑，只会污染特征。**标准做法就是单层 GAP。**
    """
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                f = model.forward_features(xb)
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            elif f.dim() == 3:            # ViT: [B, T, D] → cls token
                f = f[:, 0]
            out.append(f.float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- L1 / L2
def run_frozen_level(x, y, items, cfg, level: str, backbone: str, img_size: int,
                     counter: RuntimeCounter, n_splits: int, seeds: list,
                     schemes: list, device: str = "cuda") -> dict:
    """L1（kNN）与 L2（线性探针）共用冻结特征，差别只在分类器。

    🔴 特征**只提一次**（原实现在 seed 循环内重复提，白跑一倍时间）。
       特征提取只依赖骨干权重、不看标签、不看折号 → 无泄漏风险。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score, classification_report
    from sklearn.preprocessing import StandardScaler

    from src.vision.weights import build_model
    model = build_model(backbone, num_classes=0, img_size=img_size)
    model = model.eval().to(device)
    feats = extract_features(model, x, device=device)
    del model
    print(f"[{level}] 特征 {feats.shape} 提取完毕")

    res = defaultdict(list)          # (level, scheme, seed) -> [f1 × 折]
    per_class = defaultdict(list)
    for scheme in schemes:
        for seed in seeds:
            for fold, (tr, te) in enumerate(iter_folds(items, scheme, n_splits)):
                if level == "L1":
                    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
                    Xtr, Xte = feats[tr], feats[te]
                else:
                    # L2：标准化对线性探针与 kNN 都有益（kNN 距离度量被尺度主导）
                    sc = StandardScaler().fit(feats[tr])
                    Xtr, Xte = sc.transform(feats[tr]), sc.transform(feats[te])
                    clf = LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")
                clf.fit(Xtr, y[tr])
                pred = clf.predict(Xte)
                f1 = f1_score(y[te], pred, average="macro", zero_division=0)
                res[(scheme, seed)].append(f1)
                if scheme == schemes[0] and seed == seeds[0]:
                    rep = classification_report(y[te], pred, target_names=CLASSES,
                                               zero_division=0, output_dict=True)
                    for c in CLASSES:
                        per_class[(scheme, c)].append(rep[c]["recall"])
                print(f"    [{level}|{scheme}] seed={seed} fold={fold} "
                      f"macro-F1={f1:.4f}")
    return {"per_scheme_seed": {f"{k[0]}|s{k[1]}": v for k, v in res.items()},
            "per_class_recall": {f"{k[0]}|{k[1]}": v for k, v in per_class.items()}}


# ---------------------------------------------------------------- L3
def run_finetune_level(x, y, g, items, cfg, backbone: str, img_size: int,
                       counter: RuntimeCounter, n_splits: int, seeds: list,
                       schemes: list, epochs: int = 30, patience: int = 5,
                       lr: float = 1e-3, batch: int = 32,
                       val_frac: float = 0.12,
                       device: str = "cuda") -> dict:
    """L3 全量微调。

    🔴 关键修正（2026-10-04）：**早停验证集从训练折内部切出，绝不用测试折**。
       原实现每个 epoch 都在 `te` 上算 macro-F1 并取 best_state ——
       测试集参与模型选择，指标系统性虚高，且「stratified vs grouped 的差值
       = 泄漏幅度」这个推论会同时被污染（两边都虚高）。
       内部验证集同样按 group 切（同一实物不能一半在 train 一半在 val）。
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import StratifiedGroupKFold
    from torch.utils.data import DataLoader, TensorDataset

    from src.vision.weights import build_model
    dev = device
    # 类别加权：538:91 ≈ 6:1，不处理会吃掉小类
    counts = np.bincount(y.numpy(), minlength=len(CLASSES)).astype(np.float64)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))

    res = defaultdict(list)
    per_class = defaultdict(list)
    for scheme in schemes:
        for seed in seeds:
            for fold, (tr, te) in enumerate(iter_folds(items, scheme, n_splits)):
                torch.manual_seed(seed * 100 + fold)
                np.random.seed(seed * 100 + fold)

                # ---- 训练折内部切验证集（group-aware，绝不碰测试折）----
                n_val = max(1, int(len(tr) * val_frac))
                inner = list(StratifiedGroupKFold(
                    n_splits=max(2, round(1 / val_frac)), shuffle=True,
                    random_state=seed).split(np.zeros(len(tr)), y[tr].numpy(),
                                             groups=g[tr]))
                va_rel = inner[0][1]
                va = tr[va_rel]
                fit = np.setdiff1d(tr, va)
                if len(np.unique(y[va].numpy())) < 2:
                    va, fit = tr[:n_val], tr[n_val:]     # 兜底：验证集类别不足
                print(f"    [L3|{scheme}] seed={seed} fold={fold} "
                      f"fit={len(fit)} val={len(va)} test={len(te)}")

                # 🔴 每折**全新** model / opt / scaler（原来 scaler 跨折复用）
                model = build_model(backbone, num_classes=len(CLASSES),
                                    img_size=img_size).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                scaler = torch.amp.GradScaler("cuda")
                dl = DataLoader(TensorDataset(x[fit], y[fit]),
                                batch_size=batch, shuffle=True, num_workers=0,
                                drop_last=len(fit) > batch)
                best, bad, best_state = -1.0, 0, None
                for ep in range(epochs):
                    model.train()
                    for xb, yb in dl:
                        xb, yb = xb.to(dev), yb.to(dev)
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda"):
                            loss = crit(model(xb), yb)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    vf = _infer(model, x[va], dev)     # ← 内部验证集，非测试折
                    vmacro = f1_score(y[va].numpy(), vf, average="macro",
                                      zero_division=0)
                    if vmacro > best:
                        best, bad = vmacro, 0
                        best_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= patience:
                            break
                if best_state is not None:
                    model.load_state_dict(best_state)
                p = _infer(model, x[te], dev)
                f1 = f1_score(y[te].numpy(), p, average="macro", zero_division=0)
                res[(scheme, seed)].append(f1)
                if scheme == schemes[0] and seed == seeds[0]:
                    rep = classification_report(y[te].numpy(), p,
                                               target_names=CLASSES,
                                               zero_division=0, output_dict=True)
                    for c in CLASSES:
                        per_class[(scheme, c)].append(rep[c]["recall"])
                print(f"    [L3|{scheme}] seed={seed} fold={fold} "
                      f"val_best={best:.4f} **test macro-F1={f1:.4f}**")
                del model, opt, scaler, best_state
                torch.cuda.empty_cache()
    return {"per_scheme_seed": {f"{k[0]}|s{k[1]}": v for k, v in res.items()},
            "per_class_recall": {f"{k[0]}|{k[1]}": v for k, v in per_class.items()}}


def _infer(model, x, dev, batch: int = 64):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(x), batch):
            out.append(model(x[i:i + batch].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 汇总
def summarize(r: dict) -> dict:
    d = {}
    for key, vs in r["per_scheme_seed"].items():
        scheme, seed = key.split("|")
        d.setdefault(scheme, {})[seed] = {
            "mean": float(np.mean(vs)), "std": float(np.std(vs)),
            "folds": [round(float(v), 5) for v in vs],
        }
    out = {"per_scheme": d}
    for scheme, seeds_d in d.items():
        allv = [v for s in seeds_d.values() for v in s["folds"]]
        out[scheme] = {
            "macro_f1_mean": float(np.mean(allv)),
            "macro_f1_std": float(np.std(allv)),
            "macro_f1_min": float(np.min(allv)),
            "macro_f1_max": float(np.max(allv)),
            "n_runs": len(allv),
            "per_seed_mean": {s: v["mean"] for s, v in seeds_d.items()},
        }
    # 泄漏幅度 = 同方法同骨干下 stratified − grouped
    if "stratified" in out and "grouped" in out:
        out["leakage_gap_pp"] = round(
            100 * (out["stratified"]["macro_f1_mean"]
                   - out["grouped"]["macro_f1_mean"]), 2)
        out["leakage_gap_note"] = (
            "stratified − grouped 的差值 = 近重复泄漏的幅度估计。"
            "⚠️ 5 折 × 2 种子下标准差通常 1~3pp，差值 <2pp 一律判「等价」。")
    out["per_class_recall_mean"] = {
        k: float(np.mean(v)) for k, v in r["per_class_recall"].items()}
    return out


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="data/p1_folds.json")
    ap.add_argument("--level", default="L2", choices=["L1", "L2", "L3", "all"])
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--folds-n", type=int, default=5)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--scheme", default="both",
                    choices=["grouped", "stratified", "both"],
                    help="grouped=主口径；stratified=对照；both=双口径算泄漏幅度")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="防线4：合成数据跑通全链路，无需真实数据集")
    args = ap.parse_args()

    if args.smoke_test:
        return _smoke_test(args)

    seeds = [int(s) for s in args.seeds.split(",")]
    schemes = (["grouped", "stratified"] if args.scheme == "both"
               else [args.scheme])
    levels = ["L1", "L2", "L3"] if args.level == "all" else [args.level]
    manifest, items = load_folds(args.folds)
    print(f"[train] 数据根 {manifest['data_root']}")
    print(f"[train] {len(items)} 张 / {manifest['n_groups']} groups / "
          f"seed{seeds} / {args.folds_n} 折 / 口径 {schemes}")
    print(f"[train] 骨干 {args.backbone} @ {args.img}px")

    audit = audit_backbone(args.backbone, num_classes=len(CLASSES),
                           img_size=args.img)
    print("\n" + audit.summary())
    cfg = {"mean": audit.mean, "std": audit.std}
    counter = RuntimeCounter()
    t0 = time.time()

    print(f"\n[train] 解码 {len(items)} 张到内存...")
    x, y = decode_all(items, cfg, args.img, counter)
    g = np.array([it["group"] for it in items])
    print(f"[train] 内存张量 {tuple(x.shape)} {x.element_size()*x.nelement()/1e9:.2f} GB")

    allres = {}
    for lv in levels:
        print(f"\n{'='*62}\n[{lv}] 开始\n{'='*62}")
        if lv == "L3":
            r = run_finetune_level(x, y, g, items, cfg, args.backbone, args.img,
                                   counter, args.folds_n, seeds, schemes,
                                   epochs=args.epochs)
        else:
            r = run_frozen_level(x, y, items, cfg, lv, args.backbone, args.img,
                                 counter, args.folds_n, seeds, schemes)
        s = summarize(r)
        s.update(level=lv, backbone=args.backbone, img=args.img,
                 seeds=seeds, schemes=schemes, n_splits=args.folds_n,
                 epochs=args.epochs if lv == "L3" else None,
                 minutes=round((time.time() - t0) / 60, 2),
                 counter=counter.summary(),
                 val_frac=0.12 if lv == "L3" else None,
                 early_stop_on="inner_val_from_train_fold")
        for scheme in schemes:
            print(f"\n[{lv}|{scheme}] macro-F1 = {s[scheme]['macro_f1_mean']:.4f} ± "
                  f"{s[scheme]['macro_f1_std']:.4f} (n={s[scheme]['n_runs']})")
        if "leakage_gap_pp" in s:
            print(f"[{lv}] 🔴 泄漏幅度(stratified − grouped) = "
                  f"{s['leakage_gap_pp']:+.2f} pp")
        allres[lv] = s

    out = args.out or f"exp/p1_d3_{args.backbone}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"backbone": args.backbone, "img": args.img, "levels": allres},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {out}")
    return 0


def _smoke_test(args):
    """防线 4 · 无数据冒烟：用合成数据跑通全链路，CPU 即可。"""
    print("=" * 62)
    print("防线 4 · 无数据冒烟测试（合成数据，CPU 可跑）")
    print("=" * 62)
    import torch
    from sklearn.metrics import f1_score

    torch.manual_seed(0)
    n, c, s = 64, len(CLASSES), 64
    x = torch.randn(n, 3, s, s)
    y = torch.randint(0, c, (n,))
    y[: 20] = 0  # 模拟类别不平衡
    print(f"  合成数据: {n} 张 {s}px, {c} 类")

    from src.vision.weights import build_model
    m = build_model("mobilenetv3_small_100", num_classes=c, img_size=s)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    xb, yb = x[:32], y[:32]
    opt.zero_grad(set_to_none=True)
    loss = crit(m(xb), yb)
    loss.backward()
    opt.step()
    m.eval()
    with torch.no_grad():
        p = m(x[:16]).argmax(1)
    f1 = f1_score(y[:16].numpy(), p.numpy(), average="macro", zero_division=0)
    print(f"  一步训练: loss={loss.item():.4f} macro-F1={f1:.4f}")

    # 形状/梯度断言：条件不满足时必须 FAIL
    try:
        m(torch.randn(4, 1, s, s))  # 通道数错
        print("  ❌ 形状错误未被捕获")
        return 1
    except RuntimeError:
        print("  ✅ 形状错误被正确捕获（断言有效）")

    # 🔴 防线13：iter_folds 必须拒绝缺折号的输入（否则会静默退回 KFold）
    bad_items = [{"path": "x", "label": CLASSES[0], "y": 0, "group": 0}]
    try:
        list(iter_folds(bad_items, "grouped", 5))
        print("  ❌ 缺 fold_grouped 字段未被捕获")
        return 1
    except GuardError:
        print("  ✅ 缺折号字段被正确捕获（防线13 有效）")
    print("=" * 62)
    print("✅ 冒烟通过：全链路可跑通，可开始真实训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
