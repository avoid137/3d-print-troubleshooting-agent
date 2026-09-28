"""一次性修复：规范化 eval_set 的 gold（消除 kb_v1 悬空引用）。

> ⚠️ **状态：已执行完毕（2026-09-28），保留作过程证据。**
> 现在 `build_eval_set.py` 已内置 `V1_TO_V2` + 规范化 + 断言，
> 重跑 `python src/build_eval_set.py` 即可得到同样结果 —— **本脚本无需再跑**。
> 留存原因：它是「版本 A（只改 gold）与版本 B（重跑生成）md5 一致」这步验证的产出方。
> 若仍执行，对已修正文件是**幂等的**（无改动），只会在 `data/` 多留一份备份。

背景
----
`build_eval_set.py` 早期写于 kb_v1（75 条）时代，部分 gold 用了 v1 编号。
`preprocess.py` 清洗出 kb_v2（56 条）时按 MERGE_GROUPS 合并同义条目，次 id 被删，
但建集脚本未同步 → eval_set 里 7 个 gold 指向不存在的 kb_id（KB039/040/045/057/059/064/066），
判分必然 miss（evaluate.py:39 只比对 retrieved 里出现过的 kb_id）。

本脚本：
  1. 备份现有 eval_set.json
  2. 生成「版本 A」= 现有文件 + gold 规范化（**保持 id/query/category/顺序完全不变**）
     供修完 build_eval_set.py 后与「版本 B（重跑生成）」逐字节比对，验证随机行为未变。

用法：
  python exp/apply_gold_fix.py
"""
import json
import os
import shutil
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

EVAL = os.path.join(DATA, "eval_set.json")
KB_V2 = os.path.join(DATA, "kb_v2.json")


def load_v1_to_v2():
    """从 kb_v2.json 的 merged_from 字段派生 v1→v2 映射（与清洗逻辑同源，不靠人工推断）。"""
    v2 = json.load(open(KB_V2, encoding="utf-8"))
    m = {}
    for e in v2:
        grp = e.get("merged_from")
        if grp:
            for old in grp[1:]:
                m[old] = grp[0]
    return m, {e["kb_id"] for e in v2}


def normalize(gold, mapping, valid):
    out = []
    for g in gold:
        g2 = mapping.get(g, g)
        assert g2 in valid, f"gold {g} → {g2} 不在 kb_v2 中"
        if g2 not in out:
            out.append(g2)
    return out


def main():
    mapping, valid = load_v1_to_v2()
    print(f"从 merged_from 派生映射 {len(mapping)} 条：{mapping}\n")

    src = json.load(open(EVAL, encoding="utf-8"))

    # 1) 备份（带时间戳）
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(DATA, f"eval_set.backup_v1gold_{stamp}.json")
    shutil.copy2(EVAL, bak)
    print(f"已备份 → {os.path.basename(bak)}")

    # 2) 生成版本 A（只动 gold，不动顺序/其它字段）
    fixed = 0
    for e in src:
        old = e["gold"]
        new = normalize(old, mapping, valid)
        if new != old:
            fixed += 1
            print(f"  {e['id']}: {old} → {new}   | {e['query']}")
        e["gold"] = new

    # 3) 防复发断言
    bad = [e["id"] for e in src if not set(e["gold"]) <= valid]
    assert not bad, f"仍有悬空 gold：{bad}"
    print(f"\n共修正 {fixed} 条；全部 gold 现已落在 kb_v2 的 56 个 id 内 ✓")

    out_a = os.path.join(DATA, "_eval_set_versionA.json")
    with open(out_a, "w", encoding="utf-8") as f:
        json.dump(src, f, ensure_ascii=False, indent=2)
    print(f"版本 A 已写入 → {os.path.basename(out_a)}（供与重跑结果比对）")


if __name__ == "__main__":
    main()
