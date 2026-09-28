"""实验：对比 TF-IDF+SVD（现状）vs DashScope text-embedding-v3（真语义向量）。

目的：
  1. 量化真 embedding 对向量路 Recall@5 的提升
  2. 检验 embedding 余弦分数的「正/负例区分度」——这决定拒答能否做

产出物写入 exp/ 目录，不覆盖 data/ 下现有索引。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(r"D:\wb-workspace\2026-09-07-11-07-13\3D打印故障诊断Agent")
DATA = PROJ / "data"
EXP = PROJ / "exp"
sys.path.insert(0, str(PROJ / "src"))

CHUNKS = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
EVAL = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
TOP_K = 5


def get_key():
    for line in open(r"D:\zhixuan-retail-agent\backend\.env", encoding="utf-8", errors="ignore"):
        m = re.match(r"\s*VISION_API_KEY\s*=\s*(.+)", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None


def embed_batch(texts, text_type, api_key, batch=10):
    # 注意：text-embedding-v3 单次上限 10 条（不是 25），超过报 400 InvalidParameter
    import dashscope
    from dashscope import TextEmbedding
    dashscope.api_key = api_key
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        for attempt in range(3):
            resp = TextEmbedding.call(model="text-embedding-v3", input=chunk,
                                      text_type=text_type, dimension=1024)
            if resp.status_code == 200:
                d = {e["text_index"]: e["embedding"] for e in resp.output["embeddings"]}
                out.extend(d[j] for j in range(len(chunk)))
                break
            if attempt == 2:
                raise RuntimeError(f"embed fail: {resp.code} {resp.message}")
            time.sleep(1.5)
        print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(out, dtype=np.float32)


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main():
    key = get_key()
    if not key:
        print("无 Key"); return

    # ---------- 1. 文档向量 ----------
    print(f"[1/3] 编码 {len(CHUNKS)} 个 chunk ...")
    doc_texts = [c["text"] for c in CHUNKS]
    D = l2norm(embed_batch(doc_texts, "document", key))

    # ---------- 2. 查询向量 ----------
    print(f"[2/3] 编码 {len(EVAL)} 条 query ...")
    q_texts = [e["query"] for e in EVAL]
    Q = l2norm(embed_batch(q_texts, "query", key))

    # ---------- 3. 评测 ----------
    print("[3/3] 评测 ...")
    sims = Q @ D.T  # (200, 371) 余弦相似度
    pos_scores, neg_scores = [], []
    hits = 0
    pos_total = 0
    cat_hits, cat_total = {}, {}
    for i, item in enumerate(EVAL):
        order = np.argsort(-sims[i])[:TOP_K]
        got = [CHUNKS[j]["kb_id"] for j in order]
        top1 = float(sims[i][order[0]])
        gold = set(item["gold"])
        cat = item["category"]
        if gold:
            pos_total += 1
            cat_total[cat] = cat_total.get(cat, 0) + 1
            ok = any(k in gold for k in got)
            if ok:
                hits += 1
                cat_hits[cat] = cat_hits.get(cat, 0) + 1
            pos_scores.append(top1)
        else:
            neg_scores.append(top1)

    pos, neg = np.array(pos_scores), np.array(neg_scores)
    print()
    print("=" * 62)
    print("DashScope text-embedding-v3 · 向量路单独表现")
    print("=" * 62)
    print(f"Recall@5: {hits}/{pos_total} = {hits/pos_total:.4f}")
    print()
    print("各类别 Recall@5:")
    for c in ["single_defect", "multi_defect", "param_compensation", "ambiguous"]:
        t, h = cat_total.get(c, 0), cat_hits.get(c, 0)
        print(f"  {c:22}: {h}/{t} = {h/t if t else 0:.4f}")
    print()
    print("=== 余弦分数区分度（拒答可行性）===")
    print(f"正例 top-1: mean={pos.mean():.4f} min={pos.min():.4f} max={pos.max():.4f}")
    print(f"负例 top-1: mean={neg.mean():.4f} min={neg.min():.4f} max={neg.max():.4f}")
    print(f"分离间隔: {pos.min() - neg.max():+.4f}  (>0 说明存在可切分阈值)")

    # 最优阈值（最大化 F1）
    best = None
    for th in np.linspace(neg.min(), pos.max(), 200):
        tp = (neg < th).sum()          # 正确拒答
        fp = (pos < th).sum()          # 误拒正例
        fn = (neg >= th).sum()         # 漏拒
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if best is None or f1 > best[3]:
            best = (th, prec, rec, f1, tp, fp)
    th, prec, rec, f1, tp, fp = best
    print(f"最优阈值 {th:.4f}: 拒答准确率={prec:.4f} 召回={rec:.4f} F1={f1:.4f} "
          f"(正确拒 {tp}/{len(neg)}，误拒正例 {fp}/{pos_total})")

    # 保存
    np.save(EXP / "doc_emb_dashscope.npy", D)
    np.save(EXP / "query_emb_dashscope.npy", Q)
    res = {
        "method": "dashscope_text_embedding_v3",
        "vector_only_recall_at_5": round(hits / pos_total, 4),
        "cat_recall": {c: round(cat_hits.get(c, 0) / cat_total.get(c, 1), 4) for c in cat_total},
        "pos_top1": {"mean": round(float(pos.mean()), 4), "min": round(float(pos.min()), 4)},
        "neg_top1": {"mean": round(float(neg.mean()), 4), "max": round(float(neg.max()), 4)},
        "reject_best_threshold": round(float(th), 4),
        "reject_precision": round(float(prec), 4),
        "reject_recall": round(float(rec), 4),
    }
    (EXP / "exp_embedding_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存 exp/exp_embedding_result.json")


if __name__ == "__main__":
    main()
