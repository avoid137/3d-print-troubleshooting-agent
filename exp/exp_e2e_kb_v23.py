"""端到端验证 D 组：B2 kb 级索引 + qwen-plus 一遍法 + prompt v2.3（枚举纪律+证据采纳）。

目的：攻 multi_defect（33%）+ 证据采纳。控制变量——检索(B2)/模型(qwen-plus)/一遍法 全固定，
只把生成器 prompt 从 v2.2 升到 v2.3（逐条穷尽每 kb 全部根因 + 证据采纳平等 + multi-kb few-shot）。
对比 C 组（B2+一遍法+v2.2，来自 exp_e2e_kb_onepass_result.json）。

附加诊断：每题采集 answer 文本 + 正则提取模型实际采纳的 kb_id，
量化「检索 top-5 里有 gold 但模型没采纳」的跳过证据现象（证据采纳缺口）。

多变量提示：KB016 已在 v2.3 实验前补全根因，故 D 组相比 C 组除 prompt 外还有「KB016 补全」差异，
但该差异仅影响 eval_080 1 题（ambiguous，非 multi_defect），multi_defect 专项对比是干净的 prompt 控制变量。
"""
import os
import re
# ⚠️ 必须在 import config 之前显式关闭两遍法，确保本脚本全程一遍法。
os.environ["TWO_PASS"] = "0"

import sys
import json
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("D:/3D打印故障诊断Agent/src")
sys.path.insert(0, str(SRC))
ROOT = Path("D:/3D打印故障诊断Agent")

import faiss  # noqa: E402
from config import (KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL,  # noqa: E402
                    DASHSCOPE_API_KEY, TWO_PASS)  # noqa: E402
import answer_generator  # noqa: E402  (触发 v2.3 prompt)
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

assert TWO_PASS is False, f"TWO_PASS 未生效，仍是 {TWO_PASS}"
assert answer_generator.PROMPT_VERSION == "v2.3", \
    f"prompt 未更新到 v2.3，实际 {answer_generator.PROMPT_VERSION}"

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_kb_v23_result.json"
C_RESULT_PATH = ROOT / "exp" / "exp_e2e_kb_onepass_result.json"
SOLUTION_PREVIEW_CHARS = 250


# ---------------- B2 索引构建 ----------------
def build_kb_chunks(mode):
    kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
    chunks = []
    for item in kb:
        causes = item.get("causes", []) or []
        intro = (item.get("intro") or "").strip()
        head = item["problem"]
        if intro:
            head += "\n" + intro
        if mode == "b1":
            body = "\n".join(f"{i+1}. {c['cause']}" for i, c in enumerate(causes))
        else:  # b2：cause + solution
            body = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution', '')}"
                             for i, c in enumerate(causes))
        text = head + ("\n" + body if body else "")
        chunks.append({"chunk_id": item["kb_id"], "kb_id": item["kb_id"],
                       "problem": item["problem"], "type": "kb", "text": text})
    return chunks


def build_kb_index(chunks):
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    vecs = _dashscope_embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return bm25, index


def make_kb_retriever(chunks, bm25, vec_index):
    r = HybridRetriever(vector_backend="faiss")
    r.chunks = chunks
    r.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    r.bm25 = bm25
    r.vector_index = vec_index
    r.vector_meta = {"method": "dashscope", "model": EMBED_MODEL, "dim": EMBED_DIM}
    r.embed_method = "dashscope"
    return r


# ---------------- 判官 ----------------
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
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


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


def judge_case(q, gold, answer, strict_cache, lenient_cache, kb_by_id):
    ref = build_reference(gold, kb_by_id)
    skey = hashlib.md5(f"{STRICT_VERSION}::{q}::{answer[:200]}".encode("utf-8")).hexdigest()
    if skey in strict_cache:
        s_v = strict_cache[skey]["verdict"]
    else:
        s_v = call_judge(JUDGE_MODEL, STRICT_SYSTEM,
                         USER_TMPL.format(query=q, reference=ref, answer=answer)
                         ).get("verdict", "wrong")
        if s_v not in ("correct", "partial", "wrong"):
            s_v = "wrong"
        strict_cache[skey] = {"verdict": s_v}
    if s_v == "correct":
        l_v = "usable"
    else:
        lkey = hashlib.md5(f"{LENIENT_VERSION}::{q}::{answer[:200]}".encode("utf-8")).hexdigest()
        if lkey in lenient_cache:
            l_v = lenient_cache[lkey]["verdict"]
        else:
            l_v = call_judge(JUDGE_MODEL, LENIENT_SYSTEM,
                             USER_TMPL.format(query=q, reference=ref, answer=answer)
                             ).get("verdict", "unusable")
            if l_v not in ("usable", "unusable"):
                l_v = "unusable"
            lenient_cache[lkey] = {"verdict": l_v}
    return s_v, l_v


def extract_adopted_kb(answer):
    """从答案文本正则提取模型实际引用的 kb_id（依据 KBxxx / KBxxx 等）。"""
    return sorted(set(re.findall(r"KB\d{3}", answer)))


def summarize(records):
    rs = [r for r in records if r.get("in_scope") and not r.get("rejected", True)]
    sc = Counter(r.get("strict") for r in rs)
    lc = Counter(r.get("lenient") for r in rs)
    n_in = sum(1 for r in records if r.get("in_scope"))
    by_cat = defaultdict(Counter)
    for r in rs:
        by_cat[r.get("category", "?")][r.get("strict")] += 1
    return {
        "in_scope": n_in,
        "answered": len(rs),
        "strict": dict(sc),
        "strict_task_success": round(sc["correct"] / n_in, 4) if n_in else 0,
        "lenient": dict(lc),
        "lenient_usable_rate": round(lc["usable"] / n_in, 4) if n_in else 0,
        "by_category_strict": {k: dict(v) for k, v in by_cat.items()},
    }


def compare_to_c(summary_d):
    """从已存 C 组（B2+一遍法+v2.2）拉数字做对比（含 multi_defect 专项）。"""
    try:
        c = json.load(open(C_RESULT_PATH, encoding="utf-8"))
        sC = c["summary"]
        d_cat = summary_d["by_category_strict"]
        c_cat = sC["by_category_strict"]
        md_d = d_cat.get("multi_defect", {})
        md_c = c_cat.get("multi_defect", {})
        n_md = d_cat.get("multi_defect", {})
        def md_rate(d):
            cor = d.get("correct", 0)
            return round(cor / 30, 4)  # multi_defect 固定 30 题
        return {
            "D_kb_onepass_v23_strict": summary_d["strict_task_success"],
            "C_kb_onepass_v22_strict": sC["strict_task_success"],
            "delta_D_minus_C": round(summary_d["strict_task_success"] - sC["strict_task_success"], 4),
            "D_kb_onepass_v23_lenient": summary_d["lenient_usable_rate"],
            "C_kb_onepass_v22_lenient": sC["lenient_usable_rate"],
            "multi_defect_D_correct": md_d.get("correct", 0),
            "multi_defect_C_correct": md_c.get("correct", 0),
            "multi_defect_D_rate": md_rate(md_d),
            "multi_defect_C_rate": md_rate(md_c),
            "multi_defect_delta": round(md_rate(md_d) - md_rate(md_c), 4),
            "note": ("D=本实验(B2+一遍法+v2.3) 对比 C=已存(B2+一遍法+v2.2)。"
                     "multi_defect 专项为干净 prompt 控制变量（KB016 补全不影响该 30 题）。"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"无法加载 C 组基线: {e}"}


def main(limit=None):
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()

    # 实验组 D：B2 kb 级 + 一遍法 + prompt v2.3（TWO_PASS 已 False，answer_generator 已 v2.3）
    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever_D = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    skill_D = DiagnoseSkill(retriever=retriever_D, expand_parent=False)
    agent_D = GraphDiagnoseAgent(skill=skill_D)
    print(f"[init] B2 kb 块数={len(kb_b2)}；一遍法(TWO_PASS={TWO_PASS})；"
          f"prompt={answer_generator.PROMPT_VERSION}；共 {len(eval_set)} 题")

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)

    done = {}
    if RESULT_PATH.exists():
        old = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        for r in old.get("records", []):
            done[r["id"]] = r
        print(f"[resume] 已加载 {len(done)} 条历史结果")

    records = list(done.values())
    for i, item in enumerate(eval_set, 1):
        if item["id"] in done:
            continue
        q, gold, cat = item["query"], item["gold"], item["category"]
        in_scope = bool(gold)
        rec = {"id": item["id"], "query": q, "category": cat, "in_scope": in_scope}
        try:
            out = agent_D.answer(q)
            rec["rejected"] = out["rejected"]
            rec["kb_ids"] = out["kb_ids"]
            rec["answer"] = out["answer"]
            rec["adopted_kb"] = extract_adopted_kb(out["answer"])
            if in_scope and out["rejected"]:
                rec["strict"] = rec["lenient"] = "missed"
            elif not in_scope and not out["rejected"]:
                rec["strict"] = rec["lenient"] = "false_answer"
            elif not in_scope:
                rec["strict"] = rec["lenient"] = "ok_reject"
            else:
                sv, lv = judge_case(q, gold, out["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["strict"], rec["lenient"] = sv, lv
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(eval_set)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        RESULT_PATH.write_text(json.dumps(
            {"summary": None, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        if "error" not in rec:
            print(f"[{i}/{len(eval_set)}] {item['id']} "
                  f"rejected={rec.get('rejected')} strict={rec.get('strict')}/{rec.get('lenient')} "
                  f"adopted={rec.get('adopted_kb')}")


    summary = summarize(records)
    comp = compare_to_c(summary)
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "comparison": comp, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== B2 + 一遍法 + prompt v2.3（D 组）结果 ==========")
    print(json.dumps({"summary": summary, "comparison": comp}, ensure_ascii=False, indent=2))
    print(f"\n明细已写入 {RESULT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
