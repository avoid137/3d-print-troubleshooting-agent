"""P1 · 为「看图定稿」抽代表性图像（Cracking / Off_platform）

🔴 为什么必须看图（不是走过场）：
    `kb_mapping.py` 里 Cracking / Off_platform 两类标着 `pending_visual_check=True`、
    `checked_images=0`，备注写明「禁止凭想象定稿」。
    映射链路的正确性直接压在这两条上——若把 Cracking 硬映射到
    KB009（层间Delamination），下游会给出「调温度/层间粘合」的建议，
    而实际现象是「材料内部应力裂纹」，**建议方向会完全错**。
    数据早已到位（DC-02R 1,912 张），这项不能再挂。

抽样纪律：
    · **分层随机 + 固定种子**，可复算（不用 `random.sample` 的系统默认状态）。
    · 每类抽 N=20 张；Off_platform 只有 91 张，抽 20 张覆盖 22%，够看形态多样性。
    · 每格下方标注真值标签 + 文件名，便于人工核对时定位。
    · 拼图按 5 列排版，格子等大，不裁剪（裁剪会切掉裂纹/翘曲特征，破坏判读）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "exp"
N_PER_CLASS = 20
COLS = 5
CELL = 260
LABEL_H = 22
SEED = 20261005

#: 待定稿的两类（kb_mapping.py 里 pending_visual_check=True）
TARGET_CLASSES = ["Cracking", "Off_platform"]


def find_class_dir(root: Path, guess: str) -> Path:
    """目录名容错：DC-02R 里 Off_platform 实际是 Off-platform（下划线 vs 连字符）"""
    cands = [p for p in root.iterdir() if p.is_dir()]
    for p in cands:
        if p.name.lower().replace("_", "-") == guess.lower().replace("_", "-"):
            return p
    raise FileNotFoundError(f"找不到 {guess} 目录，现有：{[p.name for p in cands]}")


def main():
    # 从 folds 拿全量路径（唯一权威源，避免自己再遍历目录）
    # ⚠️ folds.json 里的 path 是**绝对路径**（数据集在 D:/datasets，不在仓库内），
    #    别对它做 relative_to(ROOT) —— 会抛 ValueError。
    d = json.loads((ROOT / "data" / "p1_folds.json").read_text(encoding="utf-8"))
    items = d["images"]
    by_cls: dict[str, list[str]] = {}
    for it in items:
        by_cls.setdefault(it["label"], []).append(it["path"])

    for cls in TARGET_CLASSES:
        paths = sorted(by_cls.get(cls, []))
        if not paths:
            print(f"[skip] {cls}: folds.json 里没有")
            continue
        rng = np.random.default_rng(SEED)
        take = min(N_PER_CLASS, len(paths))
        idx = np.sort(rng.choice(len(paths), size=take, replace=False))
        picked = [paths[i] for i in idx]

        rows = int(np.ceil(take / COLS))
        canvas = Image.new("RGB", (COLS * CELL, rows * (CELL + LABEL_H)), "white")
        dr = ImageDraw.Draw(canvas)
        for k, p in enumerate(picked):
            r, c = divmod(k, COLS)
            with Image.open(p) as im:
                im2 = im.convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im2, (c * CELL, r * (CELL + LABEL_H)))
            dr.text((c * CELL + 3, r * (CELL + LABEL_H) + CELL + 4),
                    f"{k+1}. {Path(p).name[:34]}", fill="black")
        out = OUT_DIR / f"p1_lookcheck_{cls}.png"
        canvas.save(out)

        # 同时给一份文本清单，供人工核对时引用
        man = OUT_DIR / f"p1_lookcheck_{cls}.json"
        man.write_text(json.dumps(
            {"cls": cls, "seed": SEED, "n_total": len(paths),
             "n_sampled": take, "grid": f"{COLS}x{rows}",
             "note": "编号与拼图格子一一对应（1-based，行优先）",
             "items": [{"no": k + 1, "path": str(p)} for k, p in enumerate(picked)]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[产出] {out.name}（{take}/{len(paths)} 张，{COLS}×{rows}）"
              f" + {man.name}")


if __name__ == "__main__":
    sys.exit(main())
