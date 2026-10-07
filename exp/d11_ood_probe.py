"""D11 · OOD 域检测探针：把 P1 从「同域 99.73% / 跨域 6.25%」收口成
「域受控 + 域外安全阀」。

目标（不是「跨域也准」，那是学术难题）
------------------------------------
量化一层安全阀：**输入漂出训练域时能否被识别出来**。
验收指标从「跨域准确率 ≥85%」（做不到）改成：

    域外检出率（TPR）  vs  域内误报率（FPR）

方法
----
特征距离族（cosine-to-centroid / kNN / Mahalanobis），用**冻结骨干**特征
（ImageNet 预训练 efficientnet_b0，**未在 DC-02R 上训练** → 天然无泄漏）。

🔴 置信度族（max-softmax/margin/energy）已由 `exp/d11_precheck_d8_scores.py`
   预检证伪：近似 AUROC **0.4136 < 0.5**（域外 median 0.980，几乎与域内重叠）。
   本脚本默认不算它们，只在末尾打印该已知反例。

为什么用冻结特征而非分类器
--------------------------
· 冻结骨干 = 与训练无关 → 规避「部署件自证」（项目纪律：折外结论不能搬到部署件）
· 目标域 DC-04 的**缺陷/正常**标签与 DC-02R 的 5 类体系不对应，
  分类器在该域的「准确率」本就无定义；OOD 检测只需要「域内 vs 域外」，不碰标签。

运行（4060）
------------
    .venv/Scripts/python.exe exp/d11_ood_probe.py                # 全量
    .venv/Scripts/python.exe exp/d11_ood_probe.py --per-view 200 # 与 D8 同采样
    .venv/Scripts/python.exe -m exp.d11_ood_probe --dry-run      # 无 GPU 逻辑自检
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.vision.ood import OODDetector, auroc, tpr_at_fpr  # noqa: E402

# --- 数据位置（与 infer.py 保持一致） ---
ID_ROOT = Path(r"D:/datasets/fdm-defect/FDM-3D-Printing-Defect-Dataset/data")
DC04_ZIP = Path(r"D:/datasets/dc04/test_data.zip")
DC04_CATS = ("layershift_error_print", "extrusion_error_print", "no_error_print")
DEFECT_CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]

OUT_JSON = ROOT / "exp" / "d11_ood_result.json"
OUT_MD = ROOT / "exp" / "d11_ood_result.md"
PRECHECK = ROOT / "exp" / "d11_precheck_result.json"


# ------------------------------------------------------------------ 收集路径
def collect_in_domain(limit_per_class: int | None = None):
    """DC-02R 域内：返回 (paths, labels)。"""
    paths, labels = [], []
    rng = random.Random(42)
    for ci, cls in enumerate(DEFECT_CLASSES):
        d = ID_ROOT / cls
        if not d.is_dir():
            raise FileNotFoundError(f"域内目录缺失: {d}（确认数据集已落 D 盘）")
        fs = sorted([p for p in d.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if limit_per_class:
            fs = rng.sample(fs, min(limit_per_class, len(fs)))
        paths += [str(p) for p in fs]
        labels += [ci] * len(fs)
    return paths, np.array(labels)


def collect_out_domain(per_view: int | None, seed: int = 42):
    """DC-04 域外：返回 (zip内路径列表, view标签, cat标签, zip路径)。"""
    if not DC04_ZIP.exists():
        raise FileNotFoundError(f"DC-04 缺失: {DC04_ZIP}")
    rng = random.Random(seed)
    names, views, cats = [], [], []
    with zipfile.ZipFile(DC04_ZIP) as z:
        all_names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        for view in ("cropped", "full_res"):
            for cat in DC04_CATS:
                pool = [n for n in all_names if f"/{view}/{cat}/" in n]
                if per_view:
                    pool = rng.sample(pool, min(per_view, len(pool)))
                names += pool
                views += [view] * len(pool)
                cats += [cat] * len(pool)
    return names, views, cats


# ------------------------------------------------------------------ 主流程
def run(F_id, y_id, F_ood, meta_ood, det_kwargs, alpha_list):
    det = OODDetector(**det_kwargs).fit(F_id, y_id)
    s_id, s_ood = det.score(F_id), det.score(F_ood)

    methods = list(s_id.keys())
    table = {}
    for m in methods:
        a = auroc(s_ood[m], s_id[m])
        tp = {f"{int(f*100)}%": round(tpr_at_fpr(s_ood[m], s_id[m], f), 4)
              for f in alpha_list}
        table[m] = {"auroc": round(a, 4), "tpr_at_fpr": tp}

    # 选定工作点 alpha=5%：域内误报≈5%，报域外检出率
    thr = det.calibrate(s_id, alpha=0.05)
    wp = det.workpoint(s_ood, thr)
    # 实际域内误报（应≈5%）
    fpr_real = {m: round(float((s_id[m] > thr[m]).mean()), 4) for m in thr}

    # 分视图
    by_view = {}
    if meta_ood and "view" in meta_ood:
        views = np.array(meta_ood["view"])
        for v in sorted(set(views)):
            sel = views == v
            by_view[v] = {m: {"auroc": round(auroc(s_ood[m][sel], s_id[m]), 4),
                              "tpr@5fpr": round(tpr_at_fpr(s_ood[m][sel], s_id[m], 0.05), 4)}
                          for m in methods}

    # 分 DC-04 类别（缺陷 vs 正常）
    by_cat = {}
    if meta_ood and "cat" in meta_ood:
        cats = np.array(meta_ood["cat"])
        for c in sorted(set(cats)):
            sel = cats == c
            by_cat[c] = {m: {"tpr@5fpr": round(tpr_at_fpr(s_ood[m][sel], s_id[m], 0.05), 4)}
                         for m in methods}

    return {
        "n_id": int(len(F_id)), "n_ood": int(len(F_ood)),
        "methods": methods,
        "table": table,
        "workpoint_alpha_0.05": {"threshold": thr, "ood_tpr": wp, "id_fpr_real": fpr_real},
        "by_view": by_view, "by_cat": by_cat,
    }


def dry_run(det_kwargs, alpha_list):
    """无 GPU 逻辑自检：合成「域内两团 + 域外正交团」。"""
    rng = np.random.default_rng(0)
    D = 1280
    c0 = np.zeros(D); c0[0] = 1.0
    c1 = np.zeros(D); c1[1] = 1.0
    F_id = np.vstack([c0 + rng.normal(0, .05, (300, D)),
                      c1 + rng.normal(0, .05, (300, D))])
    y_id = np.array([0] * 300 + [1] * 300)
    c_ood = np.zeros(D); c_ood[5] = 1.0
    F_ood = np.vstack([c_ood + rng.normal(0, .05, (150, D)),
                       c_ood + rng.normal(0, .12, (150, D))])
    meta = {"view": ["cropped"] * 150 + ["full_res"] * 150,
            "cat": ["layershift_error_print"] * 300}
    res = run(F_id, y_id, F_ood, meta, det_kwargs, alpha_list)
    res["_dry_run"] = True
    return res


def render_md(res: dict) -> str:
    L = []
    L.append("# D11 · OOD 域检测结果\n")
    L.append(f"域内样本 n={res['n_id']}（DC-02R 冻结特征）｜域外样本 n={res['n_ood']}（DC-04）"
             + ("｜**DRY-RUN（合成数据，非真实结果）**\n" if res.get("_dry_run") else "\n"))
    L.append("## 各分数区分度\n")
    L.append("| 分数 | AUROC | TPR@FPR=1% | @5% | @10% | @20% |")
    L.append("|---|---|---|---|---|---|")
    for m, v in res["table"].items():
        tp = v["tpr_at_fpr"]
        L.append(f"| {m} | **{v['auroc']}** | {tp.get('1%','-')} | {tp.get('5%','-')} | "
                 f"{tp.get('10%','-')} | {tp.get('20%','-')} |")
    L.append("\n> AUROC 0.5=无区分度，<0.5=方向反了，>0.8 才可用。\n")
    wp = res["workpoint_alpha_0.05"]
    L.append("## 工作点（域内误报锚定 5%）\n")
    L.append("| 分数 | 阈值 | 域内实际误报 | 域外检出率 |")
    L.append("|---|---|---|---|")
    for m in wp["threshold"]:
        L.append(f"| {m} | {wp['threshold'][m]:.4f} | {wp['id_fpr_real'][m]:.1%} | "
                 f"{wp['ood_tpr'].get(m, '-')} |")
    if res.get("by_view"):
        L.append("\n## 分视图\n")
        L.append("| 视图 | 分数 | AUROC | TPR@5%FPR |")
        L.append("|---|---|---|---|")
        for v, d in res["by_view"].items():
            for m, x in d.items():
                L.append(f"| {v} | {m} | {x['auroc']} | {x['tpr@5fpr']} |")
    if res.get("by_cat"):
        L.append("\n## 分 DC-04 类别（域外检出率@5%FPR）\n")
        L.append("| 类别 | " + " | ".join(res["methods"]) + " |")
        L.append("|---" * (len(res["methods"]) + 1) + "|")
        for c, d in res["by_cat"].items():
            L.append(f"| {c} | " + " | ".join(str(d[m]["tpr@5fpr"]) for m in res["methods"]) + " |")
    L.append("\n## 已知反例（不要再用）\n")
    if PRECHECK.exists():
        pc = json.load(open(PRECHECK, encoding="utf-8"))
        L.append(f"- max-softmax 做 OOD：近似 AUROC = **{pc['auroc_maxsoftmax_approx']}**"
                 f"（域外 median {pc['ood_maxsoftmax_quantiles']['p50']} vs 域内 0.99989）"
                 f" → **方向反了，不可用**（详见 exp/d11_precheck_result.json）")
    L.append("\n## 结论口径\n")
    L.append("- 对外表述：「**受控拍摄条件下的 5 类感知层 + 域外安全阀**」，"
             "不说「通用缺陷识别」或「跨域可用」。")
    L.append("- 安全阀指标 = 域外检出率 / 域内误报率，**不是**跨域准确率。")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="合成数据自检（无 GPU）")
    ap.add_argument("--per-view", type=int, default=None,
                    help="每视图每类抽样数（默认全量；D8 用 200）")
    ap.add_argument("--per-class-id", type=int, default=None, help="域内每类抽样数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=10, help="kNN 的 k")
    ap.add_argument("--methods", default="cosine_centroid,knn,mahalanobis")
    args = ap.parse_args()

    det_kwargs = {"methods": tuple(m for m in args.methods.split(",") if m), "k": args.k}
    alpha_list = (0.01, 0.05, 0.10, 0.20)

    if args.dry_run:
        res = dry_run(det_kwargs, alpha_list)
        OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        OUT_MD.write_text(render_md(res), encoding="utf-8")
        print(render_md(res))
        return 0

    # --- 真实运行：需要 torch + 数据 ---
    from src.vision.infer import embed_paths  # lazy：本机无 torch 也能 dry-run

    id_paths, y_id = collect_in_domain(args.per_class_id)
    ood_names, ood_views, ood_cats = collect_out_domain(args.per_view)
    print(f"[d11] 域内 {len(id_paths)} 张；域外 {len(ood_names)} 张（from {DC04_ZIP.name}）")

    # 防坑 ①：域内/域外不得有同名文件（集合不重叠）
    id_names = {Path(p).name for p in id_paths}
    ood_base = {Path(n).name for n in ood_names}
    inter = id_names & ood_base
    if inter:
        raise RuntimeError(f"域内/域外文件名重叠 {len(inter)} 个（如 {list(inter)[:3]}）→ 停止")

    print("[d11] 提取域内特征 ...")
    F_id = embed_paths(id_paths, device=args.device)
    print("[d11] 提取域外特征 ...")
    # 域外从 zip 直接读（避免解压残留路径失效）
    from src.vision.infer import _embed_from_zip  # noqa: PLC2701
    with zipfile.ZipFile(DC04_ZIP) as z:
        F_ood = _embed_from_zip(z, ood_names, args.device)

    # 防坑 ②：断言数量与形状
    assert F_id.shape[0] == len(y_id), (F_id.shape, len(y_id))
    assert F_ood.shape[0] == len(ood_names), (F_ood.shape, len(ood_names))
    assert F_id.shape[1] == F_ood.shape[1], "域内/域外特征维度不一致（骨干不同？）"
    print(f"[d11] 特征形状 域内{F_id.shape} 域外{F_ood.shape}")

    res = run(F_id, y_id, F_ood, {"view": ood_views, "cat": ood_cats}, det_kwargs, alpha_list)
    res["meta"] = {"id_root": str(ID_ROOT), "dc04_zip": str(DC04_ZIP),
                   "per_view": args.per_view, "backbone": "efficientnet_b0/frozen",
                   "k": args.k, "alpha_list": list(alpha_list)}
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    print(render_md(res))
    print(f"[d11] 已落盘 {OUT_JSON.name} / {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
