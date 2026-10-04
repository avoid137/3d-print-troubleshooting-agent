"""P1 · 跨类遮挡热区相似度：五类到底在看同一块，还是各看各的？

🔴 为什么单独拆一个脚本：
    遮挡热区图是**给人看的**，报告里写「五类热区各不相同」属于主观描述，
    没法被质疑也没法被复核。必须给一个**可复算的数**。

判据（先说清，避免事后编故事）：
    若模型依赖的是**共性线索**（机器本体/构图/拍摄条件），
      → 五类热区会落在同一批块上，两两相关 r 显著为正。
    若模型依赖的是**各类专属形态**，→ r 应接近 0 甚至为负。

    r > 0.5→ 判「看共性区域」   0.3~0.5 → 不可判
    r < 0.3  → 判「各看各的」

🔴 同时算「峰值块重合度」：五类峰值若集中在同一行/列，
    那一行/列就是共性区域；峰值分散在多个行/列 = 各看各的。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "exp" / "p1_d3_occlusion.json"
OUT = ROOT / "exp" / "p1_d3_occlusion_similarity.json"


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    S = d["stats"]
    cls = list(S)
    M = {c: np.array(S[c]["rel_drop_map"], dtype=float) for c in cls}

    print("=" * 72)
    print(f"跨类遮挡热区相关（n={len(cls)} 类，网格 {M[cls[0]].shape}）")
    pairs = []
    for a, b in itertools.combinations(cls, 2):
        x, y = M[a].ravel(), M[b].ravel()
        r = float(np.corrcoef(x, y)[0, 1]) if (x.std() > 1e-9 and y.std() > 1e-9) else None
        pairs.append({"a": a, "b": b, "r": None if r is None else round(r, 3)})
        print(f"  {a:<15} vs {b:<15} r="
              f"{'n/a' if r is None else f'{r:+.3f}'}")

    valid = [p["r"] for p in pairs if p["r"] is not None]
    mean_r = float(np.mean(valid)) if valid else None

    pk = {c: np.unravel_index(int(np.argmax(M[c])), M[c].shape) for c in cls}
    rows = sorted({int(p[0]) for p in pk.values()})
    cols = sorted({int(p[1]) for p in pk.values()})

    eff = {}
    for c in cls:
        h = M[c]
        thr = 0.2 * h.max()
        eff[c] = {"n": int((h > thr).sum()),
                  "pos": [[int(r), int(cc)] for r in range(h.shape[0])
                          for cc in range(h.shape[1]) if h[r, cc] > thr]}

    print("\n" + "-" * 72)
    print("峰值块：")
    for c, p in pk.items():
        print(f"  {c:<15} (row,col)=({p[0]},{p[1]})  有效块 {eff[c]['n']:>2d}")
    print(f"  峰值行覆盖 {rows} / 峰值列覆盖 {cols}")
    print("\n有效块（> 20% 自身峰值）：")
    for c in cls:
        print(f"  {c:<15} {eff[c]['n']:>2d} 块 {eff[c]['pos']}")

    print("\n" + "=" * 72)
    if mean_r is None:
        verdict = "无法判定（热区全为常数）"
    elif mean_r > 0.5:
        verdict = "五类热区高度雷同 → 模型依赖共性区域（机器本体/构图）"
    elif mean_r > 0.3:
        verdict = "不可判（0.3~0.5 灰区）"
    else:
        verdict = ("五类热区互不相关 → 模型看的是各类专属形态，"
                   "不是靠共性区域一锅端")
    print(f"平均 r = {mean_r:+.3f}" if mean_r is not None else "平均 r = n/a")
    print(f"判定：{verdict}")
    print(f"平均有效块 = {np.mean([eff[c]['n'] for c in cls]):.1f}/16"
          f"（1~2=高度集中局部；>10=信息分布式）")

    OUT.write_text(json.dumps(
        {"mean_r": None if mean_r is None else round(mean_r, 4),
         "n_pairs": len(pairs), "pairs": pairs,
         "peak": {c: [int(pk[c][0]), int(pk[c][1])] for c in cls},
         "peak_rows": rows, "peak_cols": cols,
         "effective_blocks": eff,
         "verdict": verdict,
         "threshold": {"common": ">0.5", "undecided": "0.3~0.5",
                       "distinct": "<0.3"}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())