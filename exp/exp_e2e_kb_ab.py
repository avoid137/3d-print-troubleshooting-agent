"""端到端验证：B2 kb 级索引的检索增益是否传导到生成层（paired A/B）。

配对设计（控制生成方法不变，只换检索索引）：
  A 组（控制）：chunk 级索引 + qwen-plus + 两遍法（EXPAND_PARENT=开，=生产现状）
  B 组（实验）：B2 kb 级索引（cause+solution 全量，56 块）+ qwen-plus + 两遍法
                （EXPAND_PARENT=关，kb 块已含全部根因，无需扩展）

每题同时生成 A、B 两套答案，各自用 strict / lenient 判官判分，
重点看 B 相对 A 的 strict correct 增量，以及 partial→correct 的转化数。

复用：
  - exp_index_ab.py 的 build_kb_chunks("b2") / build_kb_index / make_kb_retriever
  - skill_diagnose.DiagnoseSkill（注入自定义 retriever）
  - agent_graph.GraphDiagnoseAgent（注入自定义 skill）
  - answer_generator.AnswerGenerator（默认 qwen-plus + 两遍法）
  - 判官逻辑摘抄自 exp_e2e_lenient.py（strict + lenient 两套）

鲁棒性：逐题 try/except；结果增量落盘（可断点续跑）；生成/判官均走磁盘缓存。
"""
import sys
import json
import re
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("D:/3D打印故障诊断Agent/src")
sys.path.insert(0, str(SRC))
ROOT = Path("D:/3D打印故障诊断Agent")

import faiss
from config import (KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL,
                    DASHSCOPE_API_KEY)  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_kb_ab_result.json"
SOLUTION_PREVIEW_CHARS = 250


# ---------------- B2 索引构建（复用 exp_index_ab.py）----------------

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


# ---------------- 判官（摘抄 exp_e2e_lenient.py）----------------

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
    """对一条答案跑 strict + lenient，返回 (strict_verdict, lenient_verdict)。"""
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


def summarize(records):
    def stats(method):
        # 容错：撞错中断的题可能没有 rejected_/strict_/lenient_ 字段，
        # 统一按 .get(..., 默认值) 处理，避免汇总阶段 KeyError。
        rs = [r for r in records
              if r.get("in_scope") and not r.get(f"rejected_{method}", True)]
        sc = Counter(r.get(f"strict_{method}") for r in rs)
        lc = Counter(r.get(f"lenient_{method}") for r in rs)
        n_in = sum(1 for r in records if r.get("in_scope"))
        by_cat = defaultdict(Counter)
        for r in records:
            if r.get("in_scope") and not r.get(f"rejected_{method}", True):
                by_cat[r.get("category", "?")][f"strict_{method}"] += 1
        return {
            "in_scope": n_in,
            "answered": len(rs),
            "strict": dict(sc),
            "strict_task_success": round(sc["correct"] / n_in, 4) if n_in else 0,
            "lenient": dict(lc),
            "lenient_usable_rate": round(lc["usable"] / n_in, 4) if n_in else 0,
            "by_category_strict": {k: dict(v) for k, v in by_cat.items()},
        }
    sA, sB = stats("A"), stats("B")

    # partial→correct 转化明细（B 相对 A 的 strict 变化）
    trans = Counter()
    for r in records:
        if not r.get("in_scope"):
            continue
        a = r.get("strict_A")
        b = r.get("strict_B")
        if a is None or b is None:
            continue
        trans[f"{a}->{b}"] += 1
    partial_to_correct = trans.get("partial->correct", 0)
    correct_to_partial = trans.get("correct->partial", 0)
    wrong_to_correct = trans.get("wrong->correct", 0)
    return {
        "A_chunk_two_pass": sA,
        "B_kb_two_pass": sB,
        "delta_strict_correct": round(sB["strict_task_success"] - sA["strict_task_success"], 4),
        "delta_lenient_usable": round(sB["lenient_usable_rate"] - sA["lenient_usable_rate"], 4),
        "transitions_A_to_B": dict(trans),
        "partial_to_correct": partial_to_correct,
        "correct_to_partial": correct_to_partial,
        "wrong_to_correct": wrong_to_correct,
        "note": ("配对实验：生成方法(两遍法/qwen-plus)固定，仅检索索引不同。"
                 "B-A 的 strict 增量即检索增益对生成层的净传导。"),
    }


def main(limit=None):
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()

    # 控制组 A：chunk 级（加载现成 faiss+bm25）
    retriever_A = HybridRetriever(vector_backend="faiss")
    skill_A = DiagnoseSkill(retriever=retriever_A, expand_parent=True)
    agent_A = GraphDiagnoseAgent(skill=skill_A)

    # 实验组 B：B2 kb 级（build_kb_chunks("b2") + 内存索引）
    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever_B = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    skill_B = DiagnoseSkill(retriever=retriever_B, expand_parent=False)
    agent_B = GraphDiagnoseAgent(skill=skill_B)
    print(f"[init] B2 kb 块数={len(kb_b2)}；A/B agent 就绪；共 {len(eval_set)} 题")

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)

    # 断点续跑：加载已有结果
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
            out_A = agent_A.answer(q)
            out_B = agent_B.answer(q)
            rec["rejected_A"], rec["rejected_B"] = out_A["rejected"], out_B["rejected"]
            rec["kb_ids_A"], rec["kb_ids_B"] = out_A["kb_ids"], out_B["kb_ids"]
            if in_scope and out_A["rejected"]:
                rec["strict_A"] = rec["lenient_A"] = "missed"
            elif not in_scope and not out_A["rejected"]:
                rec["strict_A"] = rec["lenient_A"] = "false_answer"
            elif not in_scope:
                rec["strict_A"] = rec["lenient_A"] = "ok_reject"
            else:
                sv, lv = judge_case(q, gold, out_A["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["strict_A"], rec["lenient_A"] = sv, lv
            if in_scope and out_B["rejected"]:
                rec["strict_B"] = rec["lenient_B"] = "missed"
            elif not in_scope and not out_B["rejected"]:
                rec["strict_B"] = rec["lenient_B"] = "false_answer"
            elif not in_scope:
                rec["strict_B"] = rec["lenient_B"] = "ok_reject"
            else:
                sv, lv = judge_case(q, gold, out_B["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["strict_B"], rec["lenient_B"] = sv, lv
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(eval_set)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        # 增量落盘
        RESULT_PATH.write_text(json.dumps(
            {"summary": None, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        if "error" not in rec:
            print(f"[{i}/{len(eval_set)}] {item['id']} "
                  f"A={rec.get('strict_A')}/{rec.get('lenient_A')} "
                  f"B={rec.get('strict_B')}/{rec.get('lenient_B')}")

    summary = summarize(records)
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n========== 配对 A/B 结果 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细已写入 {RESULT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
