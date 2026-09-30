"""生产级验证：回填折进 src 后的端到端表现（同时充当「零回归」安全门）。

与 exp_e2e_kb_backfill.py 的区别：
  - 生成/回填走**集成后的 src 代码**（AnswerGenerator.generate_with_backfill + agent_graph），
    即真实生产路径，而非 exp 内联副本。
  - 同一次运行内对每题都算 baseline(关回填) 与 treatment(开回填)，
    用同一套集成代码、同一索引，杜绝跨文件口径漂移。
  - baseline 的生成/判官均已缓存（免费）；新增 LLM 调用 = 回填(~200) + treatment 重判(~被回填的题)。

判定：treatment 相对 baseline 是否 Δ>0 且 **correct 零回归**（correct→非 correct = 0）。
"""
import os
os.environ["TWO_PASS"] = "0"
os.environ["ANSWER_BACKFILL"] = "1"   # 生产默认开

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

import faiss  # noqa: E402
from config import (KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL,  # noqa: E402
                    DASHSCOPE_API_KEY, TWO_PASS, ANSWER_BACKFILL)  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402
from answer_generator import AnswerGenerator  # noqa: E402

assert TWO_PASS is False and ANSWER_BACKFILL is True

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_kb_backfill_prod_result.json"
SOLUTION_PREVIEW_CHARS = 250


def build_kb_chunks(mode):
    kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
    chunks = []
    for item in kb:
        causes = item.get("causes", []) or []
        head = item["problem"]
        intro = (item.get("intro") or "").strip()
        if intro:
            head += "\n" + intro
        if mode == "b1":
            body = "\n".join(f"{i+1}. {c['cause']}" for i, c in enumerate(causes))
        else:
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


def call_model(model, messages, retries=3, temperature=0.0):
    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(model=model, messages=messages,
                                   result_format="message", temperature=temperature)
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"模型调用失败: {last_err}")


def judge_case(q, gold, answer, strict_cache, lenient_cache, kb_by_id):
    ref = build_reference(gold, kb_by_id)
    skey = hashlib.md5(f"{STRICT_VERSION}::{q}::{answer[:200]}".encode("utf-8")).hexdigest()
    if skey in strict_cache:
        s_v = strict_cache[skey]["verdict"]
    else:
        s_v = call_model(JUDGE_MODEL, [
            {"role": "system", "content": STRICT_SYSTEM},
            {"role": "user", "content": USER_TMPL.format(query=q, reference=ref, answer=answer)}
        ])
        try:
            s_v = _extract_json(s_v).get("verdict", "wrong")
        except Exception:
            s_v = "wrong"
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
            l_v = call_model(JUDGE_MODEL, [
                {"role": "system", "content": LENIENT_SYSTEM},
                {"role": "user", "content": USER_TMPL.format(query=q, reference=ref, answer=answer)}
            ])
            try:
                l_v = _extract_json(l_v).get("verdict", "unusable")
            except Exception:
                l_v = "unusable"
            if l_v not in ("usable", "unusable"):
                l_v = "unusable"
            lenient_cache[lkey] = {"verdict": l_v}
    return s_v, l_v


def main(limit=None):
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    eval_set = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()
    gold_by = {e["id"]: e["gold"] for e in eval_set}

    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    skill = DiagnoseSkill(retriever=retriever, expand_parent=False)
    agent = GraphDiagnoseAgent(skill=skill)
    generator = AnswerGenerator()
    print(f"[init] B2 块数={len(kb_b2)}；回填=开；共 {len(eval_set)} 题")

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)

    done = {}
    if RESULT_PATH.exists():
        old = json.load(open(RESULT_PATH, encoding="utf-8"))
        for r in old.get("records", []):
            done[r["id"]] = r
        print(f"[resume] 已加载 {len(done)} 条")

    records = list(done.values())
    for i, item in enumerate(eval_set, 1):
        if item["id"] in done:
            continue
        q, gold, cat = item["query"], item["gold"], item["category"]
        in_scope = bool(gold)
        rec = {"id": item["id"], "query": q, "category": cat, "in_scope": in_scope}
        try:
            out = agent.answer(q)   # 集成路径：生成 + 回填（开）
            rec["rejected"] = out["rejected"]
            rec["kb_ids"] = out["kb_ids"]
            rec["backfilled"] = out.get("trace", [{}])[-1].get("backfilled", False) if out.get("trace") else False
            if in_scope and out["rejected"]:
                rec["baseline_strict"] = rec["lenient"] = rec["strict"] = "missed"
            elif not in_scope and not out["rejected"]:
                rec["baseline_strict"] = rec["lenient"] = rec["strict"] = "false_answer"
            elif not in_scope:
                rec["baseline_strict"] = rec["lenient"] = rec["strict"] = "ok_reject"
            else:
                # treatment = 集成回填答案
                tsv, tlv = judge_case(q, gold, out["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["strict"] = tsv
                rec["lenient"] = tlv
                # baseline = 关回填的草拟答案（独立生成调用，走缓存免费）
                base_out = generator.generate(q, out.get("evidence", []), top_k=5, expand=True)
                bsv, blv = judge_case(q, gold, base_out["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["baseline_strict"] = bsv
                rec["baseline_lenient"] = blv
                rec["delta"] = (tsv != bsv)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(eval_set)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        RESULT_PATH.write_text(json.dumps(
            {"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        if "error" not in rec:
            print(f"[{i}/{len(eval_set)}] {item['id']} base={rec.get('baseline_strict')}"
                  f" -> treat={rec.get('strict')} backfilled={rec.get('backfilled')}")

    # ---- 汇总 ----
    in_scope = [r for r in records if r.get("in_scope") and not r.get("rejected", True)]
    n = len(in_scope)
    base_c = Counter(r.get("baseline_strict") for r in in_scope)
    treat_c = Counter(r.get("strict") for r in in_scope)
    regress = [r["id"] for r in in_scope if r.get("baseline_strict") == "correct" and r.get("strict") != "correct"]
    improve = [r["id"] for r in in_scope if r.get("baseline_strict") != "correct" and r.get("strict") == "correct"]
    base_rate = round(base_c["correct"] / n, 4)
    treat_rate = round(treat_c["correct"] / n, 4)
    summary = {
        "n_in_scope": n,
        "baseline_strict": dict(base_c), "baseline_rate": base_rate,
        "treatment_strict": dict(treat_c), "treatment_rate": treat_rate,
        "delta_pp": round((treat_rate - base_rate) * 100, 2),
        "correct_regress": len(regress), "regress_ids": regress,
        "correct_improve": len(improve), "improve_ids": improve,
    }
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== 生产级回填（集成 src）基线 vs 处理 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
