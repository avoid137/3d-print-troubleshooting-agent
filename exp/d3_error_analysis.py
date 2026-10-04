"""P1 · 99.73% 到底错在哪：混淆矩阵 + 错误案例导出 + 难度分层

🔴 为什么必须做这一步：
    99.73% macro-F1 比文献报道的同类任务（~87%）高出约 12pp。
    数字好看不等于效果好看——**macro-F1 会掩盖「错得集中还是错得分散」**：
    · 若错误集中在 1~2 个类对→ 存在系统性混淆，落地时会成片失败
    · 若错误均匀散在各类边界 → 属正常难例，泛化风险低
    且必须**看见具体的错例**，否则无法判断是模型弱还是数据本身歧义。

产出：
    exp/p1_d3_error_analysis.json   混淆矩阵 / 逐类指标 / 错误清单
    exp/p1_errors_*.png              错误案例拼图（人眼可判）

用法：python exp/d3_error_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exp" / "p1_d3_error_analysis.json"
PNG_DIR = ROOT / "exp"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
IMG = 224
TH = 200


def main():
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    y = np.array([CLASSES.index(x["label"]) for x in items])
    g = np.array([x["group"] for x in items])
    f = np.array([x["fold_grouped"] for x in items])     # 主口径

    import torch
    from PIL import Image as _I
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 classification_report)
    from sklearn.preprocessing import StandardScaler
    from src.vision.weights import build_model
    from src.vision.guards import audit_backbone

    audit = audit_backbone("efficientnet_b0", num_classes=len(CLASSES), img_size=IMG)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = build_model("efficientnet_b0", num_classes=0, img_size=IMG).eval().cuda()

    # ---- 提特征（与 train.py 同口径：Resize((224,224)) + Normalize）----
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    cache = ROOT / "exp" / "_cache" / "l2_feats.npy"
    if cache.exists():
        F = np.load(cache)
        print(f"[err] 复用特征 {F.shape}")
    else:
        feats = []
        with torch.no_grad():
            for i in range(0, len(items), 64):
                xs = []
                for it in items[i:i + 64]:
                    with _I.open(it["path"]) as im:
                        xs.append(tf(im.convert("RGB")))
                xb = torch.stack(xs).cuda()
                with torch.amp.autocast("cuda"):
                    fr = model.forward_features(xb)
                feats.append(fr.float().mean(dim=(2, 3)).cpu())
        F = torch.cat(feats).numpy()
        np.save(cache, F)
        print(f"[err] 特征 {F.shape} → 缓存")

    # ---- 复现 L2 口径（5 折 group-aware）→ 拿混淆矩阵与错例索引 ----
    all_p, all_y, all_idx = [], [], []
    for k in range(5):
        te = np.flatnonzero(f == k)
        tr = np.flatnonzero(f != k)
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        all_p.append(clf.predict(sc.transform(F[te])))
        all_y.append(y[te])
        all_idx.append(te)
    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    IDX = np.concatenate(all_idx)
    macro = f1_score(Y, P, average="macro", zero_division=0)
    print(f"[err] 重跑 macro-F1 = {macro:.4f}（应与 train.py 的 99.73% 一致）")

    cm = confusion_matrix(Y, P)
    rep = classification_report(Y, P, target_names=CLASSES, zero_division=0,
                               output_dict=True)

    print("\n" + "=" * 66)
    print("混淆矩阵（行=真值，列=预测）")
    print(f"{'真值\\预测':<16}" + "".join(f"{c[:8]:>10}" for c in CLASSES) + f"{'recall':>10}")
    for i, c in enumerate(CLASSES):
        row = cm[i]
        rec = row[i] / max(row.sum(), 1)
        print(f"{c:<16}" + "".join(f"{v:>10}" for v in row) + f"{rec:>10.4f}")

    err_mask = P != Y
    errs = [(int(IDX[i]), int(Y[i]), int(P[i]), float(F[IDX[i]] @ F[IDX[i]]))
            for i in np.flatnonzero(err_mask)]
    print(f"\n错例总数 {len(errs)} / {len(Y)} = {len(errs)/len(Y):.2%}"
          f"（逐图准确率 {1-len(errs)/len(Y):.2%}）")

    # ---- 错得集中还是分散 ----
    pairs = {}
    for _, yt, yp, _ in errs:
        pairs[f"{CLASSES[yt]}→{CLASSES[yp]}"] = \
            pairs.get(f"{CLASSES[yt]}→{CLASSES[yp]}", 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])
    print("\n错误类对分布：")
    for k, v in top:
        print(f"  {k:<36} {v:>4}  ({v/len(errs):.1%} of errors)")
    n_pairs = len(top)
    top1_share = top[0][1] / len(errs) if top else 0
    print(f"\n  涉及{n_pairs} 个类对｜最大单对占比 {top1_share:.1%}")
    verdict = ("✅ 错误分散在多个类对 → 属边界难例，落地风险可控"
               if n_pairs >= 3 and top1_share < 0.5 else
               "⚠️ 错误集中 → 存在系统性混淆，需针对性补数据")
    print(f"  {verdict}")

    # ---- 分组折 vs 普通折：错例是否集中在被拆散的 group ----
    gs = np.array([x["group"] for x in items])
    split_gs = {gg for gg in np.unique(gs)
                if len(set(f[gs == gg])) > 1}
    n_err_split = sum(1 for i, _, _, _ in errs if gs[i] in split_gs)
    print(f"\n错例落在「被拆散 group」上的比例: {n_err_split}/{len(errs)}"
          f"（group-aware 下这类图只在训练折出现，本就不该泄漏）")

    # ---- 导出错误案例拼图 ----
    pngs = []
    if errs:
        errs_sorted = sorted(errs, key=lambda t: (t[1], t[2]))
        per = max(1, len(errs_sorted) // 4)
        for part in range(4):
            chunk = errs_sorted[part * per:(part + 1) * per]
            if not chunk:
                break
            cols = min(6, len(chunk))
            rows = (len(chunk) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * TH, rows * (TH + 16)), "white")
            dr = ImageDraw.Draw(canvas)
            for j, (gi, yt, yp, _) in enumerate(chunk):
                r, c = j // cols, j % cols
                with Image.open(items[gi]["path"]) as im:
                    canvas.paste(im.convert("RGB").resize((TH, TH)),
                                 (c * TH, r * (TH + 16)))
                dr.text((c * TH + 2, r * (TH + 16) + 2),
                        f"T:{CLASSES[yt][:9]} P:{CLASSES[yp][:9]}", fill="red")
            p = PNG_DIR / f"p1_errors_part{part}.png"
            canvas.save(p)
            pngs.append(p.name)
            print(f"[err] 导出 {p.name}（{len(chunk)} 例）")

    out = {
        "scheme": "grouped (主口径)", "n": len(Y),
        "macro_f1_rerun": round(float(macro), 4),
        "n_errors": len(errs),
        "per_image_acc": round(1 - len(errs) / len(Y), 4),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "per_class": {c: {k: round(float(rep[c][k]), 4)
                          for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "error_pairs": top,
        "n_error_pairs": n_pairs,
        "top_pair_share": round(float(top1_share), 4),
        "verdict": verdict,
        "error_pngs": pngs,
        "error_examples": [
            {"file": Path(items[gi]["path"]).name, "truth": CLASSES[yt],
             "pred": CLASSES[yp]} for gi, yt, yp, _ in errs[:25]],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
