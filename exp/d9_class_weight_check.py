# -*- coding: utf-8 -*-
"""D9 收尾 · 分类器口径核对：`class_weight` 到底用不用 balanced？

为什么要专门核对
------------------------------------------------------------------
落地 6 类部署件时发现一个**未验证的口径不一致**：

  | 位置 | 参数 |
  |---|---|
  | `exp/d9_ab_normal_class.py`（A/B 实测，得出「加类有效」的那个实验） | `max_iter=4000, C=1.0`，**无** class_weight |
  | `src/vision/infer.py`（部署件）+ `train.py`（99.73% 那个） | `max_iter=3000, C=1.0, class_weight="balanced"` |

这不是小差别。6 类体系里正常类只有 200 张，缺陷类最多 538 张：
`balanced` 会给正常类约 **(1912+200)/6/200 ≈ 1.76 倍**… 实际上按 sklearn 的
`n_samples / (n_classes * n_count)` 算，正常类权重被放大到约 **5.0×**，
而 538 张的 Warping 只有 0.63×。**正常类的决策边界会被显著推高**，
直接决定「宁可误报还是宁可漏报」。

🔴 若不核对就重建部署件，等于**用一套没跑过 A/B 的参数去落地 A/B 的结论**。
   这正是本项目反复吃过亏的坑（P0「模型选择误用测试折」的同类）。

本脚本只回答一个问题，不新增实验组
------------------------------------------------------------------
在 **N=200、6 类、group-aware 折外** 同一设定下，
`class_weight=None` 与 `"balanced"` 的三个关键指标差多少：

  ① DC-02R 5 类 macro-F1（不能因为加权重就崩）
  ② DC-04 正常件正确识别率（加类的**全部意义**就在这里）
  ③ 缺陷件保留率（不能为了压误报把缺陷件也吃成No_error）

只用**折外**预测算（项目铁律：部署件/训练集上的指标是自证）。

用法
------------------------------------------------------------------
    .venv/Scripts/python.exe -u exp/d9_class_weight_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "exp"))
from d9_ab_normal_class import (  # noqa: E402
    CLASSES_5, CLASSES_6, FEAT_CACHE, FOLDS, fit_predict, load_dc02r,
    load_dc04_eval, macro_f1, DC04_TRAIN_CLASS,
)

#: 折外特征缓存：DC-04 正常件（N=200）提一次约 40s，不缓存则每种口径都重提一遍
NORMAL_FEAT_CACHE = ROOT / "exp" / "_cache" / "dc04_normal_n200.npz"
OUT_JSON = ROOT / "exp" / "d9_classweight_check.json"

N = 200
SEED = 42
NRM = len(CLASSES_5)          # 正常类索引 = 5


def _load_normal_feats(device: str) -> tuple:
    """DC-04 正常件特征（带缓存）。场景划分由 load_dc04_normal 内部保证。"""
    if NORMAL_FEAT_CACHE.exists():
        z = np.load(NORMAL_FEAT_CACHE, allow_pickle=True)
        print(f"[cache] 命中 {NORMAL_FEAT_CACHE.name}  shape={z['X'].shape}")
        return z["X"].astype(np.float64), z["scenes"]

    import d9_ab_normal_class as d9
    X, scenes = d9.load_dc04_normal(N, SEED, device)
    NORMAL_FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NORMAL_FEAT_CACHE, X=X.astype(np.float32),
                        scenes=np.array(scenes, dtype=object))
    print(f"[cache] 已写入 {NORMAL_FEAT_CACHE.name}")
    return X, scenes


def _patch_class_weight(mode: str):
    """把 fit_predict 内部的 LogReg 换成指定 class_weight。

    🔴 用**包装**而不是改d9_ab_normal_class.py 源码：A/B 的历史结果 json
       必须保持「默认权重」口径可复现，不能被本次核对改写。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import d9_ab_normal_class as d9

    def fit_predict_cw(Xtr, ytr, Xte, full_classes=None):
        kw = {} if mode == "none" else {"class_weight": mode}
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=1.0, **kw))
        pipe.fit(Xtr, ytr)
        seen = np.asarray(pipe[-1].classes_)
        pred = np.asarray(pipe.predict(Xte), dtype=int)
        pred = seen[np.clip(pred, 0, len(seen) - 1)] if full_classes is not None else pred
        proba = pipe.predict_proba(Xte)
        if proba.ndim == 1:
            proba = np.repeat(proba[:, None], len(seen), axis=1)
        if full_classes is not None:
            n_c = len(list(full_classes))
            aligned = np.zeros((len(Xte), n_c))
            for j, c in enumerate(seen):
                aligned[:, int(c)] = proba[:, j]
            proba = aligned
        return pred, proba, pipe

    orig = d9.fit_predict
    d9.fit_predict = fit_predict_cw
    return orig


def run_one(mode: str, X02, y02, fold02, Xn, Xe, is_normal) -> dict:
    """在指定 class_weight 下跑 N=200 / 6 类 / 折外评估。"""
    import d9_ab_normal_class as d9
    orig = _patch_class_weight(mode)
    try:
        yn = np.full(len(Xn), NRM)
        y_all = np.concatenate([y02, yn])

        # DC-04 正常件按**场景**切 5 折（与 A/B 同法，避免同场景跨折泄漏）
        rng = np.random.RandomState(SEED)
        nf = rng.randint(0, 5, len(yn))

        oof = np.full(len(y_all), -1)
        oof_proba = np.zeros((len(y_all), len(CLASSES_6)))
        for k in range(5):
            tr02, te02 = fold02 != k, fold02 == k
            trn = nf != k
            Xtr = np.concatenate([X02[tr02], Xn[trn]])
            ytr = np.concatenate([y02[tr02], yn[trn]])
            p, pb, _ = d9.fit_predict(
                Xtr, ytr, np.concatenate([X02[te02], Xn[nf == k]]),
                full_classes=CLASSES_6)
            sel = np.concatenate([te02, nf == k])
            oof[sel] = p
            oof_proba[sel] = pb
        assert (oof != -1).all(), "有样本未被任何折覆盖"

        # ① DC-02R 5 类
        f1_5, per5 = macro_f1(y_all[:len(y02)], oof[:len(y02)], len(CLASSES_5))
        # ②③ 折外：正常件正确 / 缺陷件保留（阈值 0.90，与部署一致）
        conf = oof_proba.max(1)
        answered = conf >= 0.90
        is_nrm = np.concatenate([np.zeros(len(y02), bool), np.ones(len(yn), bool)])
        n_ok = float(((oof == NRM) & is_nrm).sum()) / max(int(is_nrm.sum()), 1)
        d_ok = float((answered & (oof != NRM) & (~is_nrm)).sum()) / \
            max(int((~is_nrm).sum()), 1)
        return {
            "class_weight": mode,
            "dc02r_macro_f1_5": f1_5,
            "per_f1_5": dict(zip(CLASSES_5, per5)),
            "normal_correct": n_ok,
            "defect_preserved_th090": d_ok,
        }
    finally:
        d9.fit_predict = orig


def main() -> int:
    device = "cuda"
    X02, y02, fold02 = load_dc02r()
    print(f"[DC-02R] {X02.shape}  类分布 "
          f"{ {c: int((y02 == i).sum()) for i, c in enumerate(CLASSES_5)} }")

    print("\n[DC-04 正常件] 提特征…")
    Xn, _ = _load_normal_feats(device)

    print("\n[DC-04 试金石] 提特征…")
    Xe, te = load_dc04_eval(device, SEED)
    is_normal = np.array([t == DC04_TRAIN_CLASS for t in te])

    print("\n" + "=" * 68)
    print(f"口径核对：N={N} /6 类 / group-aware 折外")
    print("=" * 68)
    res = {}
    for mode in ["none", "balanced"]:
        r = run_one(mode, X02, y02, fold02, Xn, Xe, is_normal)
        res[mode] = r
        print(f"\n  class_weight={mode!r}")
        print(f"    DC-02R 5 类 macro-F1 = {r['dc02r_macro_f1_5']*100:.2f}%")
        print(f"    正常件正确率(n={N})   = {r['normal_correct']*100:.1f}%")
        print(f"    缺陷件保留率(th=0.90) = {r['defect_preserved_th090']*100:.1f}%")

    d5 = (res["balanced"]["dc02r_macro_f1_5"] - res["none"]["dc02r_macro_f1_5"]) * 100
    dn = (res["balanced"]["normal_correct"] - res["none"]["normal_correct"]) * 100
    dd = (res["balanced"]["defect_preserved_th090"]
          - res["none"]["defect_preserved_th090"]) * 100
    print("\n" + "-" * 68)
    print("balanced − none的差值：")
    print(f"    DC-02R 5 类 macro-F1{d5:+.2f}pp")
    print(f"    正常件正确率           {dn:+.1f}pp")
    print(f"    缺陷件保留率           {dd:+.1f}pp")
    print("-" * 68)

    # 判定规则先写死，避免看到结果再挑说法
    keep = []
    if (res["none"]["dc02r_macro_f1_5"] >= 0.99
            and res["none"]["normal_correct"] >= 0.99):
        keep.append("none")
    if (res["balanced"]["dc02r_macro_f1_5"] >= 0.99
            and res["balanced"]["normal_correct"] >= 0.99):
        keep.append("balanced")
    verdict = ("none" if "none" in keep and "balanced" not in keep
               else "balanced" if "balanced" in keep and "none" not in keep
               else "两者均可（差异在噪声内），取与 A/B 实测一致者" if keep
               else "两者都不达标 → 不要落地，先查数据")
    res["_delta_balanced_minus_none_pp"] = {
        "dc02r_macro_f1_5": round(d5, 3),
        "normal_correct": round(dn, 3),
        "defect_preserved": round(dd, 3),
    }
    res["_verdict"] = verdict
    print(f"→ 建议部署口径：**{verdict}**")

    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[产出] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())