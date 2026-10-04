"""判官可靠性审计：用更强判官（qwen-plus）重判生产答案，检验 54.37% 是否被偏严判官低估。

动机：端到端 strict 主指标由 qwen-turbo 判官给出。若判官系统性偏严（如把"表述不同但实质正确"
判为 partial/wrong），则真实水平被低估；反之若偏松则被高估。两者都必须如实报告。

方法：
  1. 生产答案从缓存重建（generate_with_backfill 命中 answer_cache/backfill_cache → 零生成成本）
  2. 同一批答案交给 qwen-plus 判官，用**完全相同**的 strict prompt 重判
  3. 对比两个判官的 verdict 分布与逐题一致率

⚠️ 这不是"刷分"：换判官不改系统，只检验测量工具。若两判官分歧大，说明指标本身有测量噪声，
   结论必须双向报告（plus 更严也要如实写）。

用法：
    python exp/exp_judge_audit.py            # 全量 160
    python exp/exp_judge_audit.py --limit 8  # 冒烟
产物：exp/exp_judge_audit_result.json + exp/judge_audit_result.md
"""
import sys
import json
import time
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

# 复用 prod 实验的构建逻辑与判官 prompt（保证口径完全一致）
from exp_e2e_kb_backfill_prod import (  # noqa: E402
    STRICT_SYSTEM, USER_TMPL, STRICT_VERSION, EVAL_SET_PATH, DASHSCOPE_API_KEY,
    build_kb_chunks, build_kb_index, make_kb_retriever, load_kb, build_reference,
    _extract_json, call_model,
)
from config import KB_V2_PATH  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

AUDIT_JUDGE = "qwen-plus-2025-07-28"
BASE_JUDGE = "qwen-turbo"
RESULT_PATH = ROOT / "exp" / "exp_judge_audit_result.json"
AUDIT_CACHE = ROOT / "data" / "judge_audit_cache.json"
PROD_RESULT = ROOT / "exp" / "exp_e2e_kb_backfill_prod_result.json"


def load_cache(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def judge_with(model, q, gold, answer, kb_by_id, cache, tag):
    ref = build_reference(gold, kb_by_id)
    key = hashlib.md5(f"{tag}::{model}::{STRICT_VERSION}::{q}::{answer[:200]}"
                      .encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]["verdict"]
    raw = call_model(model, [
        {"role": "system", "content": STRICT_SYSTEM},
        {"role": "user", "content": USER_TMPL.format(query=q, reference=ref, answer=answer)},
    ], retries=5)
    try:
        v = _extract_json(raw).get("verdict", "wrong")
    except Exception:
        v = "wrong"
    if v not in ("correct", "partial", "wrong"):
        v = "wrong"
    cache[key] = {"verdict": v}
    return v


def main(limit=None):
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    eval_set = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
    prod = json.load(open(PROD_RESULT, encoding="utf-8"))
    turbo_by = {r["id"]: r for r in prod["records"]}
    kb_by_id = load_kb()

    items = [e for e in eval_set
             if turbo_by.get(e["id"], {}).get("in_scope")
             and not turbo_by.get(e["id"], {}).get("rejected")]
    if limit:
        items = items[:limit]
    print(f"[init] 审计题目数={len(items)}；基准判官={BASE_JUDGE}；审计判官={AUDIT_JUDGE}")

    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    agent = GraphDiagnoseAgent(skill=DiagnoseSkill(retriever=retriever, expand_parent=False))

    cache = load_cache(AUDIT_CACHE)
    done = {}
    if RESULT_PATH.exists():
        for r in json.load(open(RESULT_PATH, encoding="utf-8")).get("records", []):
            if "error" not in r:
                done[r["id"]] = r
        print(f"[resume] 已加载 {len(done)} 条")
    records = list(done.values())

    for i, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        q, gold = item["query"], item["gold"]
        rec = {"id": item["id"], "query": q, "category": item["category"],
               "turbo_strict": turbo_by[item["id"]].get("strict")}
        try:
            out = agent.answer(q)          # 缓存命中 → 零生成成本
            rec["answer"] = out["answer"]
            rec["kb_ids"] = out["kb_ids"]
            rec["plus_strict"] = judge_with(
                AUDIT_JUDGE, q, gold, out["answer"], kb_by_id, cache, "audit")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(items)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        RESULT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        AUDIT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if "error" not in rec:
            print(f"[{i}/{len(items)}] {item['id']} turbo={rec['turbo_strict']} "
                  f"plus={rec.get('plus_strict')}", flush=True)

    # ---- 汇总 ----
    clean = [r for r in records if "error" not in r]
    t = Counter(r["turbo_strict"] for r in clean)
    p = Counter(r["plus_strict"] for r in clean)
    n = len(clean)
    agree = sum(1 for r in clean if r["turbo_strict"] == r["plus_strict"])
    matrix = Counter((r["turbo_strict"], r["plus_strict"]) for r in clean)
    tur_correct = t.get("correct", 0)
    plus_correct = p.get("correct", 0)
    summary = {
        "n": n,
        "turbo_dist": dict(t), "plus_dist": dict(p),
        "turbo_correct_rate": round(tur_correct / n, 4),
        "plus_correct_rate": round(plus_correct / n, 4),
        "delta_pp": round((plus_correct - tur_correct) / n * 100, 2),
        "agreement": round(agree / n, 4),
        "matrix": {f"{a}->{b}": c for (a, b), c in sorted(matrix.items())},
        "turbo_correct_plus_not": [r["id"] for r in clean
                                   if r["turbo_strict"] == "correct" and r["plus_strict"] != "correct"],
        "turbo_not_plus_correct": [r["id"] for r in clean
                                   if r["turbo_strict"] != "correct" and r["plus_strict"] == "correct"],
    }
    RESULT_PATH.write_text(json.dumps({"summary": summary, "records": records},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 判官审计汇总 =====")
    print(f"样本 n={n}")
    print(f"  qwen-turbo 分布: {dict(t)}  correct率 {summary['turbo_correct_rate']:.2%}")
    print(f"  qwen-plus  分布: {dict(p)}  correct率 {summary['plus_correct_rate']:.2%}")
    print(f"  Δ(plus - turbo) = {summary['delta_pp']} pp")
    print(f"  一致率 = {summary['agreement']:.2%}")
    print(f"  转化矩阵: {summary['matrix']}")
    print(f"  turbo=correct 但 plus≠correct: {len(summary['turbo_correct_plus_not'])} 题")
    print(f"  turbo≠correct 但 plus=correct: {len(summary['turbo_not_plus_correct'])} 题")


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(lim)
