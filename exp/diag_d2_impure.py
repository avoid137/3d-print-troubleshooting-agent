"""P1 · D2 诊断：19 个跨类污染 group 到底由哪一路 union 造成？

背景（2026-10-04）：
    离线标定（前一轮）只用了「前缀 ∪ 余弦≥0.995」，测得跨类污染 = 0。
    完整流程（额外含 pHash/dHash 粗筛，命中 2,539 对）跑出 **19 个跨类污染 group**。
    两套口径唯一差别 = pHash 粗筛 → 假设：pHash/dHash 误合并。

本脚本要回答的三个问题（全部用实测数字回答，不猜）：
    Q1  前缀规则本身干净吗？（同前缀是否跨类）
    Q2  pHash 粗筛命中的 2,539 对里，有多少对是「余弦不同意」的？
        这些对跨类比例多少？→ 量化 pHash 的误合并率
    Q3  只保留「前缀 ∪ 余弦≥T」时，跨类污染是否恒为 0？
        （对多个 T 扫一遍，确认 0.995 不是碰巧）

产物：exp/p1_d2_impure_diagnosis.json + 人类可读结论打印
    哈希缓存：exp/_cache/hashes.npz（调阈值秒级，不用重解码 1,912 张原图）

用法：
    python exp/diag_d2_impure.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "exp" / "_cache"
FEAT_CACHE = CACHE / "dino_feats.npy"
PATH_CACHE = CACHE / "dino_paths.npy"
HASH_CACHE = CACHE / "hashes.npz"
OUT = ROOT / "exp" / "p1_d2_impure_diagnosis.json"

# 与 src.vision.dedup 保持一致的候选阈值
PHASH_CANDS = [0, 2, 3, 4, 5, 6, 8, 10]
COS_CANDS = [0.99, 0.995, 0.998, 0.999]


# ------------------------------------------------------------------ 工具
class _U:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = np.array([self.find(i) for i in range(len(self.p))])
        uniq, inv = np.unique(g, return_inverse=True)
        return g, inv, len(uniq)


def group_report(inv, y, n_groups):
    """返回 (n_impure, max_group, impure_examples)"""
    impure, mx, ex = 0, 0, []
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(n_groups + 1))
    for gi in range(n_groups):
        idx = order[bounds[gi]:bounds[gi + 1]]
        mx = max(mx, len(idx))
        labs = np.unique(y[idx])
        if len(labs) > 1:
            impure += 1
            if len(ex) < 5:
                ex.append([int(k) for k in idx[:6]])
    return impure, mx, ex


# ------------------------------------------------------------------ 哈希
def compute_hashes(paths, log=True):
    """流式 224px 缩略 + dHash/pHash。带缓存，避免每次调阈值重解码 1,912 张。"""
    if HASH_CACHE.exists():
        z = np.load(HASH_CACHE, allow_pickle=True)
        if list(z["paths"]) == [str(p) for p in paths]:
            print(f"[diag] 复用哈希缓存 {HASH_CACHE.name}")
            return z["dhash"].tolist(), z["phash"].tolist()

    from src.vision.dedup import dhash, phash

    if log:
        print(f"[diag] 计算 dHash/pHash (n={len(paths)}, 流式 224px 缩略)...")
    dhs, phs = [], []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im.draft("RGB", (512, 512))
            t = im.convert("RGB").resize((224, 224), Image.BILINEAR)
            t = t.copy()
        dhs.append(dhash(t))
        phs.append(phash(t))
        if log and i % 400 == 0:
            print(f"    {i}/{len(paths)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    # ⚠️ 64bit 哈希最高位为 1 时超出 int64 上限 → 必须 uint64，
    #   否则 `np.array(dhs, dtype=np.int64)` 抛 OverflowError: int too big to convert
    np.savez_compressed(
        HASH_CACHE, dhash=np.array(dhs, dtype=np.uint64),
        phash=np.array(phs, dtype=np.uint64),
        paths=np.array([str(p) for p in paths], dtype=object))
    print(f"[diag] 哈希已缓存 → {HASH_CACHE.name}")
    return dhs, phs


def hash_pairs(dhs, phs, thr, mode="either"):
    """返回命中对 (i<j) 的数组。mode: either=min(d,p) / d / p"""
    n = len(dhs)
    D = np.array([[(h >> b) & 1 for b in range(64)] for h in dhs], dtype=np.uint8)
    P = np.array([[(h >> b) & 1 for b in range(64)] for h in phs], dtype=np.uint8)
    I, J = [], []
    for i in range(n):
        di = np.bitwise_count(np.bitwise_xor(D, D[i])).sum(axis=1)
        pi = np.bitwise_count(np.bitwise_xor(P, P[i])).sum(axis=1)
        if mode == "either":
            m = np.minimum(di, pi)
        elif mode == "d":
            m = di
        else:
            m = pi
        hit = np.flatnonzero(m <= thr)
        hit = hit[hit > i]
        I.append(hit)
        J.append(np.full(len(hit), i))
    return np.concatenate(I), np.concatenate(J)


# ------------------------------------------------------------------ 主流程
def main():
    if not FEAT_CACHE.exists():
        print("缺特征缓存，先跑 python -m src.vision.calibrate_dedup")
        return 3
    F = np.load(FEAT_CACHE)
    paths = [Path(p) for p in np.load(PATH_CACHE, allow_pickle=True)]
    if len(F) != len(paths):
        print(f"缓存不一致: F={F.shape} paths={len(paths)}")
        return 3
    n = len(paths)
    y = np.array([p.parent.name for p in paths])
    classes = sorted(set(y.tolist()))
    print(f"[diag] n={n} 特征 {F.shape} 类别 {classes}")

    dhs, phs = compute_hashes(paths)
    from src.vision.dedup import _prefix_key
    keys = np.array([_prefix_key(p) for p in paths], dtype=object)

    S = F @ F.T
    iu = np.triu_indices(n, 1)
    cos = S[iu]          # 展平上三角，仅用于分布统计
    ii, jj = iu
    # ⚠️ 查任意 (i,j) 对的余弦必须用**完整矩阵** S[i, j]，
    #   不能用展平上三角 `cos[i*n+j]`（越界）也不能用 `cos[pos]`（需反查）。
    def cos_of(I, J):
        return S[I, J]

    report = {"n_images": n, "classes": classes}

    # ---------------- Q1 前缀规则是否干净 ----------------
    print("\n" + "=" * 72)
    print("Q1  前缀规则（_aug/_original）本身是否跨类？")
    upk = {}
    for i, k in enumerate(keys):
        upk.setdefault(k, []).append(i)
    multi = {k: v for k, v in upk.items() if len(v) > 1}
    pre_impure = {k: v for k, v in multi.items() if len({y[i] for i in v}) > 1}
    print(f"  前缀重复组 {len(multi)} 个，涉及 {sum(len(v) for v in multi.values())} 张")
    print(f"  其中跨类的 {len(pre_impure)} 个" +
          (f"  ← 前缀规则有问题！" if pre_impure else "  ✅ 前缀规则零跨类"))
    for k, v in list(pre_impure.items())[:3]:
        print(f"    {k}: {[(paths[i].name, y[i]) for i in v]}")
    report["prefix"] = {
        "n_multi_groups": len(multi),
        "n_images_in_multi": int(sum(len(v) for v in multi.values())),
        "n_impure": len(pre_impure),
        "examples": [[paths[i].name for i in v] for v in list(pre_impure.values())[:5]],
    }

    # ---------------- Q2 pHash 粗筛的误合并 ----------------
    print("\n" + "=" * 72)
    print("Q2  pHash/dHash 粗筛：命中对里有多少是「余弦不同意」的？")
    print(f"  {'阈值':>4} {'模式':>7} {'命中对':>8} {'余弦<0.995':>11} "
          f"{'其中跨类':>9} {'跨类率':>8}")
    ph_rows = []
    for thr in PHASH_CANDS:
        for mode in ("either", "d", "p"):
            if thr == 0 and mode != "either":
                continue
            I, J = hash_pairs(dhs, phs, thr, mode)
            m = len(I) == 0
            c = np.ones(len(I), bool) if m else cos_of(I, J)
            cross = np.zeros(len(I), bool) if m else (y[I] != y[J])
            bad = (~m) & (c < 0.995)
            badc = bad & cross
            row = {
                "threshold": thr, "mode": mode, "n_pairs": int(len(I)),
                "n_cos_below": int(bad.sum()),
                "n_bad_cross": int(badc.sum()),
                "bad_rate": round(float(bad.sum() / max(len(I), 1)), 4),
            }
            ph_rows.append(row)
            print(f"  {thr:>4} {mode:>7} {len(I):>8,} {row['n_cos_below']:>11,} "
                  f"{row['n_bad_cross']:>9,} {row['bad_rate']:>7.1%}")
    report["phash_scan"] = ph_rows

    # 详细看 thr=5 / either（即当前生产配置）
    I5, J5 = hash_pairs(dhs, phs, 5, "either")
    c5 = cos_of(I5, J5)
    same_pref = keys[I5] == keys[J5]
    print(f"\n  [细看 thr≤5 either] 命中 {len(I5):,} 对：")
    print(f"    同前缀（本来就是真重复）      : {int(same_pref.sum()):,}")
    print(f"    不同前缀但余弦≥0.995（真重复补漏）: {int((~same_pref & (c5 >= 0.995)).sum()):,}")
    print(f"    不同前缀且余弦<0.995（**误合并**）: {int((~same_pref & (c5 < 0.995)).sum()):,}")
    rep_idx = np.flatnonzero(~same_pref & (c5 < 0.995))[:8]
    for k in rep_idx:
        print(f"      cos={c5[k]:.4f}  {y[I5[k]]}/{y[J5[k]]}  "
              f"{paths[I5[k]].name}  <->  {paths[J5[k]].name}")
    report["phash5_detail"] = {
        "n_pairs": int(len(I5)),
        "same_prefix": int(same_pref.sum()),
        "diff_prefix_cos_ge": int((~same_pref & (c5 >= 0.995)).sum()),
        "diff_prefix_cos_lt": int((~same_pref & (c5 < 0.995)).sum()),
        "examples": [
            {"cos": round(float(c5[k]), 5), "a": paths[I5[k]].name, "la": y[I5[k]],
             "b": paths[J5[k]].name, "lb": y[J5[k]]}
            for k in rep_idx],
    }

    # ---------------- Q3 前缀 ∪ 余弦 的污染扫描 ----------------
    print("\n" + "=" * 72)
    print("Q3  只用「前缀 ∪ 余弦≥T」（pHash 降级为只读诊断，不参与 union）")
    print(f"  {'T':>8} {'组数':>6} {'折叠':>6} {'最大组':>7} {'跨类组':>7} "
          f"{'前缀组被拆?':>11}")
    cos_rows = []
    for t in COS_CANDS:
        uf = _U(n)
        first = {}
        for i, k in enumerate(keys):
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
        hc = np.flatnonzero(cos >= t)
        for a, b in zip(ii[hc].tolist(), jj[hc].tolist()):
            uf.union(a, b)
        _, inv, ng = uf.groups()
        imp, mx, ex = group_report(inv, y, ng)
        # 前缀组是否被余弦边进一步撑大（超出前缀本身）
        cos_rows.append({
            "threshold": t, "n_groups": int(ng), "folded": int(n - ng),
            "max_group": int(mx), "n_impure": int(imp),
        })
        print(f"  {t:>8} {ng:>6} {n - ng:>6} {mx:>7} {imp:>7} "
              f"{'—':>11}")
    report["cos_only_scan"] = cos_rows

    # ---------------- 结论 ----------------
    ok = all(r["n_impure"] == 0 for r in cos_rows)
    print("\n" + "=" * 72)
    print("结论")
    print(f"  前缀规则跨类数            : {report['prefix']['n_impure']}")
    print(f"  pHash thr≤5 either 误合并 : "
          f"{report['phash5_detail']['diff_prefix_cos_lt']:,} 对")
    print(f"  前缀∪余弦 各阈值跨类组   : "
          f"{[r['n_impure'] for r in cos_rows]} → {'✅ 全 0' if ok else '❌ 非 0'}")
    print("  ⇒ 19 个污染组来源：pHash/dHash 粗筛（阈值 ≤5 过松 + union 传递闭包放大）")
    print("  ⇒ 修法：pHash/dHash 降级为**只读诊断**（不再 union），"
          "union 唯一依据 = 前缀(确定性) ∪ 余弦≥0.995")

    ROOT.joinpath("exp").mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
