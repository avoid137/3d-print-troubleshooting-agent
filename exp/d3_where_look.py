"""P1 · 模型在看哪：裁剪带对照（Where-does-it-look）

🔴 触发原因：
    遮挡热图画出来后发现一个比分数更要紧的事实——
    DC-02R 的构图是**整台打印机 + 底下一小块打印件**，
    缺陷件在画面里占比很小。那么 99.73% 到底是
 「看懂了缺陷件」还是「看懂了机器/构图/件在画面里的位置」？
    遮挡热区只能给相关信号，不能给因果结论。做**因果对照**：

    把图像横向切三带，各自单独送进同一个 L2 流程：
      band=top    (0%~33% 高度)   → 上部：打印机风扇/标牌/背景
      band=mid   (33%~66%)        → 中部：热床/打印头
      band=bottom(66%~100%)       → 下部：**缺陷件所在带**
      band=full  基准=必须复现 99.73%
    若 bottom 带单独就能拿到高分 → 模型确实在读缺陷件。
    若只有 full 才高分 → 高分来自跨带的整体上下文（构图/位置），
    换拍摄距离就失效，落地风险高。

🔴 纪律：
  1. **full 档必须复现 L2 的 99.73%**（±0.1pp），否则整份对照作废并直接报错。
     不是「差不多就行」——管线不一致的话，band 之间的差值没有可比性。
  2. 裁剪是**先切原图再 resize**，等于放大该带；不是 resize 后切，
     后者会丢掉放大效应，与真实「只看这一带」不等价。
  3. 复用 D2 的 fold_grouped，不重做 KFold。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exp" / "p1_d3_where_look.json"
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
IMG = 224
N_SPLITS = 5
ANCHOR_PP = 0.1          # full 档与 L2 基准的允许偏差
L2_BASELINE = 0.9973    # exp/p1_vision_result.md 记录的 L2 / B0 主口径

BANDS = {
    "full": (0.00, 1.00),
    "top": (0.00, 0.33),
    "mid": (0.33, 0.66),
    "bottom": (0.66, 1.00),
    "bottom_half": (0.50, 1.00),
}


def extract(paths, y0, y1, dev="cuda"):
    """先按比例切原图，再 resize 到 IMG×IMG"""
    import torch
    from PIL import Image
    from torchvision import transforms
    from src.vision.weights import build_model
    tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406),
                                                  (0.229, 0.224, 0.225))])
    model = build_model("efficientnet_b0", num_classes=0, img_size=IMG).eval().to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), 64):
            xs = []
            for p in paths[s:s + 64]:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    xs.append(tf(im.crop((0, int(H * y0), W, int(H * y1)))))
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                fr = model.forward_features(torch.stack(xs).to(dev))
            out.append(fr.float().mean(dim=(2, 3)).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=list(range(len(CLASSES)))))


def l2_eval(F, y, folds, seeds=(42, 43)):
    """与 train.py 同配方：StandardScaler + LogReg(balanced)，逐折 fit，绝不跨折复用"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scores = []
    for k in range(N_SPLITS):
        te = np.flatnonzero(folds == k)
        tr = np.flatnonzero(folds != k)
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        scores.append(macro_f1(y[te], clf.predict(sc.transform(F[te]))))
    return float(np.mean(scores)), float(np.std(scores)), scores


def main():
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    paths = [str(ROOT / x["path"]) for x in items]
    y = np.array([CLASSES.index(x["label"]) for x in items])
    folds = np.array([x["fold_grouped"] for x in items])

    print("=" * 72)
    print("裁剪带对照：模型到底在读哪一条带？")
    print(f"基准 full 档必须落在 {L2_BASELINE*100:.2f}% ± {ANCHOR_PP*100:.1f}pp，否则整份对照作废")
    res = {}
    for name, (y0, y1) in BANDS.items():
        F = extract(paths, y0, y1)
        m, s, per = l2_eval(F, y, folds)
        res[name] = {"band": [y0, y1], "macro_f1_mean": round(m, 5),
                     "macro_f1_std": round(s, 5),
                     "per_fold": [round(v, 5) for v in per]}
        print(f"  {name:<12} y=[{y0:.2f},{y1:.2f}]  "
              f"macro-F1 = {m*100:.2f}% ± {s*100:.2f}%")

    anchor = res["full"]["macro_f1_mean"]
    dev_pp = abs(anchor - L2_BASELINE) * 100
    print("\n" + "-" * 72)
    print(f"基准校验：full = {anchor*100:.2f}%vs 记录值 {L2_BASELINE*100:.2f}%"
          f"（差 {dev_pp:.2f}pp，阈值 {ANCHOR_PP*100:.1f}pp）")
    if dev_pp > ANCHOR_PP:
        print("  ❌ 管线不一致，本次对照作废，不出结论")
        OUT.write_text(json.dumps({"status": "INVALID", "anchor": anchor,
                                   "expected": L2_BASELINE},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    print("  ✅ 一致，带间差值可比")

    print("\n关键对比（相对 full）：")
    for name in ("top", "mid", "bottom", "bottom_half"):
        d_pp = (res[name]["macro_f1_mean"] - anchor) * 100
        print(f"  {name:<12} {res[name]['macro_f1_mean']*100:>6.2f}%   "
              f"{d_pp:+6.2f}pp")

    # 判据：底部带能否独立成立
    bot = res["bottom_half"]["macro_f1_mean"] * 100
    top = res["top"]["macro_f1_mean"] * 100
    print("\n" + "=" * 72)
    print("判读：")
    if bot >= 90:
        print(f"  ✅ 下半部（缺陷件所在带）单独就有 {bot:.2f}%")
        print("     → 模型确实在读缺陷件，高分不是靠构图/机器外观撑起来的")
    elif bot >= 70:
        print(f"  ⚠️ 下半部单独只有 {bot:.2f}%（full {anchor*100:.2f}%）")
        print("     → 模型部分依赖跨带上下文，换拍摄距离会掉分")
    else:
        print(f"  ❌ 下半部单独仅 {bot:.2f}%，而 full {anchor*100:.2f}%")
        print("     → 高分主要来自跨带整体上下文（构图/位置），落地风险高")
    if top >= bot:
        print(f"  ⚠️ 上部（{top:.2f}%）竟不低于下部（{bot:.2f}%）——"
              "机器外观/背景可能承载了类别信息，需警惕")

    OUT.write_text(json.dumps(
        {"n_splits": N_SPLITS, "classifier": "L2 (StandardScaler+LogReg)",
         "anchor_expected": L2_BASELINE, "anchor_dev_pp": round(dev_pp, 3),
         "results": res,
         "verdict": {"bottom_half_pp": round(bot, 2), "top_pp": round(top, 2),
                     "full_pp": round(anchor * 100, 2)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())