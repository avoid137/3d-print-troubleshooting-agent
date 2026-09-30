"""v2.8b 混合：v2.8 枚举式草稿 + append-only 回填保底 —— 全量安全门实验。

动机：v2.8 单独用因 Call2 skipped 丢 gold 产生 23 道回归（零回归安全门未过）。
本实验在 v2.8 草稿后接**确定性 append-only 回填**（只补证据里相关但草稿漏掉的根因，不删不改），
预期：保留 v2.8 的 27 道救回 + 用回填补回跳过的 gold（回收回归）+ 回填经证零回归。

对比基线：
  - 回填基线（生产 prod 结果）：安全门判定对象（regress 须=0 方可折进主路）
  - v2.8 基线（exp_e2e_v28_result.json）：验证回填不丢 v2.8 已得 correct（append-only 应 ~0 回归）
做法：检索走集成 src，生成用 generate_v28_with_backfill，判官复用缓存。
脚本可续跑：done 剔除含 error 项以便重试；call_model 重试 5×退避 2s 抗代理 502。
"""
import os
os.environ["TWO_PASS"] = "0"
os.environ["ANSWER_BACKFILL"] = "1"

import sys
import json
import time
import re
import hashlib
from collections import Counter
from pathlib import Path

SRC = Path("D:/3D打印故障诊断Agent/src")
sys.path.insert(0, str(SRC))
ROOT = Path("D:/3D打印故障诊断Agent")

import faiss  # noqa: E402
from config import (KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL,  # noqa: E402
                    DASHSCOPE_API_KEY)  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402
from answer_generator import AnswerGenerator  # noqa: E402

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
PROD_RESULT = ROOT / "exp" / "exp_e2e_kb_backfill_prod_result.json"
V28_RESULT = ROOT / "exp" / "exp_e2e_v28_result.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_v28b_result.json"
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


def call_model(model, messages, retries=5, temperature=0.0):
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
        time.sleep(2.0 * (attempt + 1))
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

    prod = json.load(open(PROD_RESULT, encoding="utf-8"))
    prod_strict = {r["id"]: r.get("strict") for r in prod.get("records", [])}
    prod_lenient = {r["id"]: r.get("lenient") for r in prod.get("records", [])}
    v28 = json.load(open(V28_RESULT, encoding="utf-8"))
    v28_strict = {r["id"]: r.get("v28_strict") for r in v28.get("records", [])}
    v28_lenient = {r["id"]: r.get("v28_lenient") for r in v28.get("records", [])}

    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    skill = DiagnoseSkill(retriever=retriever, expand_parent=False)
    agent = GraphDiagnoseAgent(skill=skill)
    generator = AnswerGenerator()
    print(f"[init] B2 块数={len(kb_b2)}；v2.8b 混合生成；共 {len(eval_set)} 题")

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)

    done = {}
    if RESULT_PATH.exists():
        old = json.load(open(RESULT_PATH, encoding="utf-8"))
        for r in old.get("records", []):
            if "error" not in r:
                done[r["id"]] = r
        print(f"[resume] 已完成 {len(done)} 条（error 项将重试）")

    records = list(done.values())
    for i, item in enumerate(eval_set, 1):
        if item["id"] in done:
            continue
        q, gold, cat = item["query"], item["gold"], item["category"]
        in_scope = bool(gold)
        rec = {"id": item["id"], "query": q, "category": cat, "in_scope": in_scope}
        try:
            out = agent.answer(q)
            evidence = out.get("evidence", [])
            rec["rejected"] = out["rejected"]
            rec["kb_ids"] = out["kb_ids"]
            rec["backfill_strict"] = prod_strict.get(item["id"])
            rec["backfill_lenient"] = prod_lenient.get(item["id"])
            rec["v28_strict"] = v28_strict.get(item["id"])
            rec["v28_lenient"] = v28_lenient.get(item["id"])
            if in_scope and out["rejected"]:
                rec["v28b_strict"] = rec["v28b_lenient"] = "missed"
            elif not in_scope and not out["rejected"]:
                rec["v28b_strict"] = rec["v28b_lenient"] = "false_answer"
            elif not in_scope:
                rec["v28b_strict"] = rec["v28b_lenient"] = "ok_reject"
            else:
                g = generator.generate_v28_with_backfill(q, evidence, top_k=5, expand=True)
                rec["v28b_answer"] = g["answer"]
                rec["v28b_backfilled"] = g.get("backfilled", False)
                rec["v28b_skip"] = g.get("skipped", [])
                rec["v28b_cached"] = g.get("cached", False)
                tsv, tlv = judge_case(q, gold, g["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["v28b_strict"] = tsv
                rec["v28b_lenient"] = tlv
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(eval_set)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        RESULT_PATH.write_text(json.dumps(
            {"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        if "error" not in rec:
            print(f"[{i}/{len(eval_set)}] {item['id']} backfill={rec.get('backfill_strict')}"
                  f" v28={rec.get('v28_strict')} -> v28b={rec.get('v28b_strict')}"
                  f" bf={rec.get('v28b_backfilled')} cached={rec.get('v28b_cached')}")

    # ---- 汇总 ----
    in_scope = [r for r in records if r.get("in_scope") and not r.get("rejected", True)]
    n = len(in_scope)
    bf_c = Counter(r.get("backfill_strict") for r in in_scope)
    v28_c = Counter(r.get("v28_strict") for r in in_scope)
    v28b_c = Counter(r.get("v28b_strict") for r in in_scope)
    reg_vs_bf = [r["id"] for r in in_scope
                 if r.get("backfill_strict") == "correct" and r.get("v28b_strict") != "correct"]
    imp_vs_bf = [r["id"] for r in in_scope
                 if r.get("backfill_strict") != "correct" and r.get("v28b_strict") == "correct"]
    reg_vs_v28 = [r["id"] for r in in_scope
                  if r.get("v28_strict") == "correct" and r.get("v28b_strict") != "correct"]
    imp_vs_v28 = [r["id"] for r in in_scope
                  if r.get("v28_strict") != "correct" and r.get("v28b_strict") == "correct"]
    bf_rate = round(bf_c["correct"] / n, 4)
    v28_rate = round(v28_c["correct"] / n, 4)
    v28b_rate = round(v28b_c["correct"] / n, 4)

    by_cat = {}
    for r in in_scope:
        c = r["category"]
        by_cat.setdefault(c, {"n": 0, "bf_correct": 0, "v28_correct": 0, "v28b_correct": 0})
        by_cat[c]["n"] += 1
        if r.get("backfill_strict") == "correct":
            by_cat[c]["bf_correct"] += 1
        if r.get("v28_strict") == "correct":
            by_cat[c]["v28_correct"] += 1
        if r.get("v28b_strict") == "correct":
            by_cat[c]["v28b_correct"] += 1
    for c, d in by_cat.items():
        d["bf_rate"] = round(d["bf_correct"] / d["n"], 4)
        d["v28_rate"] = round(d["v28_correct"] / d["n"], 4)
        d["v28b_rate"] = round(d["v28b_correct"] / d["n"], 4)

    bf_l = Counter(r.get("backfill_lenient") for r in in_scope)
    v28_l = Counter(r.get("v28_lenient") for r in in_scope)
    v28b_l = Counter(r.get("v28b_lenient") for r in in_scope)

    summary = {
        "n_in_scope": n,
        "backfill_strict": dict(bf_c), "backfill_rate": bf_rate,
        "v28_strict": dict(v28_c), "v28_rate": v28_rate,
        "v28b_strict": dict(v28b_c), "v28b_rate": v28b_rate,
        "delta_vs_backfill_pp": round((v28b_rate - bf_rate) * 100, 2),
        "delta_vs_v28_pp": round((v28b_rate - v28_rate) * 100, 2),
        "regress_vs_backfill": len(reg_vs_bf), "regress_vs_backfill_ids": reg_vs_bf,
        "improve_vs_backfill": len(imp_vs_bf), "improve_vs_backfill_ids": imp_vs_bf,
        "regress_vs_v28": len(reg_vs_v28), "regress_vs_v28_ids": reg_vs_v28,
        "improve_vs_v28": len(imp_vs_v28), "improve_vs_v28_ids": imp_vs_v28,
        "by_category": by_cat,
        "backfill_lenient_rate": round(bf_l.get("usable", 0) / n, 4),
        "v28_lenient_rate": round(v28_l.get("usable", 0) / n, 4),
        "v28b_lenient_rate": round(v28b_l.get("usable", 0) / n, 4),
    }
    RESULT_PATH.write_text(json.dumps(
        {"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== v2.8b 混合(v2.8草稿+回填保底) vs 回填基线 / v2.8 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
