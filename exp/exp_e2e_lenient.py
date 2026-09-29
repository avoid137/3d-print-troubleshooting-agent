"""P2 判据核查：对**同一批答案**同时跑 strict 与 lenient 两套判官。

背景：strict 判官要求「覆盖全部 gold 主要根因」，实测任务成功率仅 15%。
疑问：判据是否对「排障手册」过严？用户真实要的是「最可能的根因 + 可行对策」，
不一定穷尽所有 cause。

本脚本生成一次答案（top-1 扩展，走 answer_cache），然后：
- strict 判官（沿用 exp_e2e_eval.py 判据，结果落 judge_e2e_cache）
- lenient 判官（新增宽松判据：答案至少给出一条**与参照一致的可用根因+对策**）
用 lenient 得到「可用率」，与 strict 的「穷尽率」对照。

⚠️ 诚实披露：lenient 是**可用性视角**，不是主指标。答辩时必须说清
两个数字各自的含义，绝不把 lenient 当正式 accuracy 吹。
本脚本只回答「判据过严是否高估了真实可用率」。

lenient 判据示例：
- eval_001 'temperature tower calibration' gold=KB003+KB004
  如果答案给出 KB003 的正确根因+温度调整 → lenient=usable（虽然没穷尽 KB004）
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DASHSCOPE_API_KEY, KB_V2_PATH, EVAL_SET_PATH  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_lenient.json"
SOLUTION_PREVIEW_CHARS = 250

STRICT_SYSTEM = """你是 RAG 系统答案质量评审员。给定用户问题、标准参照
（gold 知识库条目）和系统答案，判断答案是否正确。
- correct：答案覆盖参照中**全部**主要根因，方案与参照一致。
- partial：只覆盖部分主要根因，或方案不完整。
- wrong：基本未覆盖 / 与参照矛盾 / 答非所问。
只输出 JSON：{"verdict": "correct|partial|wrong", "reason": "一句话"}"""

LENIENT_SYSTEM = """你是排障答案**可用性**评审员。用户拿到这份答案后，
是否能照着做、给出至少一条与知识库一致且可行的诊断/对策？
- usable：答案中存在**至少一条**与标准参照一致的主要根因，且对策可操作、
  与参照不冲突。（不需要穷尽所有根因。）
- unusable：答案根因整体错误、与参照矛盾、或答非所问、或对策不可操作。
只输出 JSON：{"verdict": "usable|unusable", "reason": "一句话"}"""

USER_TMPL = """用户问题：
{query}

标准参照（gold 知识库条目）：
{reference}

系统答案：
{answer}

请输出 JSON 评审结果。"""


def load_kb():
    with open(KB_V2_PATH, "r", encoding="utf-8") as f:
        return {e["kb_id"]: e for e in json.load(f)}


def build_reference(gold_kb_ids, kb_by_id):
    blocks = []
    for kb_id in gold_kb_ids:
        e = kb_by_id.get(kb_id)
        if not e:
            continue
        lines = [f"【{kb_id}】问题: {e['problem']}"]
        for c in e["causes"]:
            lines.append(f"  根因: {c['cause']}\n  方案: {c['solution'][:SOLUTION_PREVIEW_CHARS]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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


def load_cache(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path, cache):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def call_judge(model, system, user_msg, retries=3):
    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}],
                result_format="message", temperature=0.0)
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                try:
                    return _extract_json(content)
                except (json.JSONDecodeError, AttributeError):
                    return {"reason": "parse-fail"}
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"judge 调用失败: {last_err}")


def main(limit=None):
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()
    agent = GraphDiagnoseAgent()

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)

    records, used = [], 0
    for i, item in enumerate(eval_set, 1):
        q, gold = item["query"], item["gold"]
        in_scope = bool(gold)
        # 答案：走 answer_cache，各罚扩展已回退 top-1
        out = agent.answer(q)
        answer = out["answer"]
        rec = {"id": item["id"], "query": q, "category": item["category"],
               "in_scope": in_scope, "rejected": out["rejected"]}
        if not in_scope:
            rec["strict"] = "ok_reject" if out["rejected"] else "false_answer"
            rec["lenient"] = "ok_reject" if out["rejected"] else "false_answer"
            records.append(rec)
            continue
        if out["rejected"]:
            rec["strict"] = rec["lenient"] = "missed"
            records.append(rec)
            continue

        skey = hashlib.md5(f"{STRICT_VERSION}::{q}::{answer[:200]}"
                           .encode("utf-8")).hexdigest()
        # strict：命中缓存直接取，否则调
        if skey in strict_cache:
            s_v = strict_cache[skey]["verdict"]
        else:
            s_v = call_judge(JUDGE_MODEL, STRICT_SYSTEM,
                             USER_TMPL.format(query=q,
                                              reference=build_reference(gold, kb_by_id),
                                              answer=answer)).get("verdict", "wrong")
            strict_cache[skey] = {"verdict": s_v}
        if s_v not in ("correct", "partial", "wrong"):
            s_v = "wrong"
        rec["strict"] = s_v
        used += not strict_cache and 0

        # lenient：strict 已 correct 则必 usable，跳过省 token；否则调
        lkey = hashlib.md5(f"{LENIENT_VERSION}::{q}::{answer[:200]}"
                           .encode("utf-8")).hexdigest()
        if s_v == "correct":
            rec["lenient"] = "usable"
            rec["lenient_skipped"] = True
        elif lkey in lenient_cache:
            rec["lenient"] = lenient_cache[lkey]["verdict"]
        else:
            l_v = call_judge(JUDGE_MODEL, LENIENT_SYSTEM,
                             USER_TMPL.format(query=q,
                                              reference=build_reference(gold, kb_by_id),
                                              answer=answer)).get("verdict", "unusable")
            if l_v not in ("usable", "unusable"):
                l_v = "unusable"
            lenient_cache[lkey] = {"verdict": l_v}
            rec["lenient"] = l_v

        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        records.append(rec)
        print(f"[{i}/{len(eval_set)}] {item['id']} strict={rec['strict']} "
              f"lenient={rec['lenient']}")

    # 汇总
    sc = Counter(r["strict"] for r in records if r["in_scope"] and not r["rejected"])
    lc = Counter(r["lenient"] for r in records if r["in_scope"] and not r["rejected"])
    n_in = sum(1 for r in records if r["in_scope"])
    strict_ok = sc["correct"]
    lenient_ok = lc["usable"]
    scope_f1 = sum(1 for r in records if r["in_scope"] and not r["rejected"])

    summary = {
        "in_scope": n_in,
        "answered": scope_f1,
        "strict": dict(sc),
        "strict_task_success": round(strict_ok / n_in, 4),
        "lenient": dict(lc),
        "lenient_usable_rate": round(lenient_ok / n_in, 4),
        "delta": round((lenient_ok - strict_ok) / n_in, 4),
        "note": "lenient=可用率(非主指标)，strict=穷尽率(主指标)",
    }
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 判据核查 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {RESULT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)