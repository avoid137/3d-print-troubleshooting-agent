"""P2 端到端评测：Agent 全链路（LangGraph 版）+ LLM-as-judge 判分。

此前所有指标都是「零件级」：检索 Recall@5 = 0.85、拒答 F1 = 0.9873。
本实验第一次测「整机合格率」——用户问一句，Agent 最终答得对不对。

指标
----
1. 范围决策（全链路拒答）：Precision / Recall / F1（160 正例 + 40 负例）
   - 误拒（正例被拒）与漏拒（负例放行）分开统计
2. 答案质量（库内且未拒答的条目）：LLM-as-judge 对照 gold KB 全文判
   verdict ∈ {correct, partial, wrong}
3. 端到端任务成功率 = correct / 160（对外主指标）
4. 按类别分解：single_defect / multi_defect / param_compensation / ambiguous

判官判据（写死在 prompt 里，防止被答案的流畅度带偏）：
   correct = 覆盖全部 gold 条目的主要根因，解决方案与 KB 一致（可合理精简）
   partial = 只覆盖部分 gold 条目，或方案不完整
   wrong   = 基本未覆盖 / 与 KB 矛盾 / 答非所问
库内条目被拒答 → 直接记 missed（不进 judge），算失败。

缓存：判官结果落盘 data/judge_e2e_cache.json（key 含 JUDGE_PROMPT_VERSION
与答案文本，答案变了自动重判）。生成侧复用 answer_cache。
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DASHSCOPE_API_KEY, KB_V2_PATH, EVAL_SET_PATH  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

JUDGE_MODEL = "qwen-turbo"
JUDGE_PROMPT_VERSION = "v1"        # 改判据必须递增
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_result.json"
SOLUTION_PREVIEW_CHARS = 250       # 每个 gold solution 喂给判官的上限


# ---------------- gold 参照构建 ----------------

def load_kb():
    with open(KB_V2_PATH, "r", encoding="utf-8") as f:
        return {e["kb_id"]: e for e in json.load(f)}


def build_reference(gold_kb_ids, kb_by_id):
    """把 gold KB 条目压成判官可读的参照文本。"""
    blocks = []
    for kb_id in gold_kb_ids:
        e = kb_by_id.get(kb_id)
        if not e:
            continue
        lines = [f"【{kb_id}】问题: {e['problem']}"]
        for c in e["causes"]:
            sol = c["solution"][:SOLUTION_PREVIEW_CHARS]
            lines.append(f"  - 根因: {c['cause']}\n    方案: {sol}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------- 判官 ----------------

JUDGE_SYSTEM = """你是 RAG 系统答案质量评审员。给定用户问题、标准参照
（来自知识库 gold 条目）和系统答案，判断答案是否正确。

判据（严格执行，不被答案的流畅度/篇幅影响）：
- correct：答案覆盖了参照中**全部**主要根因，给出的解决方案与参照一致
  （允许合理精简、换措辞、合并表述）。
- partial：答案只覆盖了参照中**部分**根因，或解决方案明显不完整。
- wrong：答案基本未覆盖参照根因、与参照矛盾、或答非所问。

注意：答案引用参照之外的常识不算错，但**不能作为命中依据**。

只输出 JSON：{"verdict": "correct|partial|wrong", "reason": "一句话理由"}"""

JUDGE_USER = """用户问题：
{query}

标准参照（gold 知识库条目）：
{reference}

系统答案：
{answer}

请输出 JSON 评审结果。"""


def _extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_answer(query, reference, answer, cache, model=JUDGE_MODEL, retries=3):
    key = hashlib.md5(
        f"{JUDGE_PROMPT_VERSION}::{model}::{query}::{answer[:400]}"
        .encode("utf-8")).hexdigest()
    if key in cache:
        return {**cache[key], "cached": True}

    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    user_msg = JUDGE_USER.format(query=query, reference=reference,
                                 answer=answer)
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_msg}],
                result_format="message", temperature=0.0)
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                try:
                    data = _extract_json(content)
                    verdict = data.get("verdict", "wrong")
                    if verdict not in ("correct", "partial", "wrong"):
                        verdict = "wrong"
                except (json.JSONDecodeError, AttributeError):
                    verdict, data = "wrong", {"reason": "judge 输出解析失败"}
                entry = {"verdict": verdict,
                         "reason": str(data.get("reason", ""))[:200]}
                cache[key] = entry
                return {**entry, "cached": False}
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"judge 调用失败: {last_err}")


# ---------------- 主流程 ----------------

def main(limit=None):
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()
    agent = GraphDiagnoseAgent()
    cache = load_cache()

    records = []
    for i, item in enumerate(eval_set, 1):
        q, gold = item["query"], item["gold"]
        in_scope = bool(gold)
        out = agent.answer(q)

        rec = {"id": item["id"], "query": q, "category": item["category"],
               "in_scope": in_scope, "rejected": out["rejected"],
               "layer": out["layer"], "kb_ids": out["kb_ids"]}

        if in_scope and out["rejected"]:
            rec["verdict"] = "missed"          # 误拒：库内被拒
        elif (not in_scope) and (not out["rejected"]):
            rec["verdict"] = "false_answer"    # 漏拒：库外放行
        elif not in_scope:
            rec["verdict"] = "ok_reject"       # 正确拒答
        else:
            ref = build_reference(gold, kb_by_id)
            v = judge_answer(q, ref, out["answer"], cache)
            rec["verdict"] = v["verdict"]
            rec["judge_reason"] = v["reason"]
            if not v["cached"]:
                save_cache(cache)
        records.append(rec)
        print(f"[{i}/{len(eval_set)}] {item['id']} -> {rec['verdict']}"
              f"{' (cached)' if rec.get('cached') else ''}")

    # ---------------- 汇总 ----------------
    n_pos = sum(r["in_scope"] for r in records)
    n_neg = len(records) - n_pos
    verdicts = Counter(r["verdict"] for r in records)

    # 全链路拒答 P/R/F1（正例=库内应放行）
    tp = sum(1 for r in records if r["in_scope"] and not r["rejected"])
    fp = sum(1 for r in records if not r["in_scope"] and not r["rejected"])
    fn = sum(1 for r in records if r["in_scope"] and r["rejected"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0

    # 答案质量（仅库内且放行的条目参与判分）
    judged = [r for r in records
              if r["in_scope"] and not r["rejected"]]
    n_correct = sum(1 for r in judged if r["verdict"] == "correct")
    n_partial = sum(1 for r in judged if r["verdict"] == "partial")
    n_wrong = sum(1 for r in judged if r["verdict"] == "wrong")

    # 端到端任务成功率（对外主指标）：correct / 全部库内条目
    e2e_acc = n_correct / n_pos if n_pos else 0.0

    by_cat = defaultdict(lambda: Counter())
    for r in records:
        by_cat[r["category"]][r["verdict"]] += 1

    summary = {
        "total": len(records), "in_scope": n_pos, "out_of_scope": n_neg,
        "scope_gate": {"precision": round(prec, 4), "recall": round(rec_, 4),
                       "f1": round(f1, 4), "missed_reject": fn,
                       "false_answer": fp},
        "answer_quality": {"judged": len(judged), "correct": n_correct,
                           "partial": n_partial, "wrong": n_wrong,
                           "correct_rate_among_answered":
                               round(n_correct / len(judged), 4) if judged else 0},
        "e2e_task_success": round(e2e_acc, 4),
        "verdict_dist": dict(verdicts),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 端到端评测结果 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细已写入 {RESULT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 条（调试用）")
    args = ap.parse_args()
    main(limit=args.limit)
