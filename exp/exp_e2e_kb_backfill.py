"""端到端验证：B2 kb 级索引 + qwen-plus + 一遍法(v2.2) + 结构化「完整性回填」(v2.4 雏形)。

目的：攻击选项③瓶颈——54 题 gold 已在检索上下文(top-5)却没答对(partial/wrong)。
      做法（受控消融，不改动生产 src）：
        1. Call1 = 基线 v2.2 一遍法生成（磁盘缓存命中，零新增生成成本）
        2. 仅对「gold⊆检索top-5 且 strict∈{partial,wrong}」的题做 Call2 回填：
           让模型复核草拟答案是否覆盖证据中全部相关根因，只补「证据里已有、
           且确实相关、但答案漏掉」的根因（严禁补证据外内容）。
        3. 对回填后答案重新判官，得 treatment strict/lenient。
      对照：基线 C（exp_e2e_kb_onepass_result.json）= 79/160=49.38%。

诚实性：回填 pass 的输入只有「检索到的证据」，与运行时可用信息一致；
       不注入 gold。判官仍用 gold 评测，结果反映真实能力。

下行有界：回填只可能「多加噪声」不会「丢失已答对的 gold」；
           correct 题不动，故 treatment 最坏 = 基线，不会拖垮。
"""
import os
os.environ["TWO_PASS"] = "0"   # 继承基线一遍法

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
                    DASHSCOPE_API_KEY, TWO_PASS)  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

assert TWO_PASS is False

JUDGE_MODEL = "qwen-turbo"
BACKFILL_MODEL = "qwen-plus-2025-07-28"   # 回填质量敏感，用与生成同档
STRICT_VERSION = "v1"
LENIENT_VERSION = "v1"
BASELINE_RESULT = ROOT / "exp" / "exp_e2e_kb_onepass_result.json"
CACHE_PATH = ROOT / "data" / "judge_e2e_cache.json"
LENIENT_PATH = ROOT / "data" / "judge_lenient_cache.json"
BACKFILL_CACHE = ROOT / "data" / "backfill_cache.json"
RESULT_PATH = ROOT / "exp" / "exp_e2e_kb_backfill_result.json"
SOLUTION_PREVIEW_CHARS = 250


# ---------------- B2 索引构建（同 onepass） ----------------
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


# ---------------- 判官（同 onepass） ----------------
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


# ---------------- 回填 pass ----------------
BACKFILL_SYSTEM = """你是 FDM 3D 打印故障诊断的「完整性复核员」。
你会拿到：①用户问题 ②系统已检索到的知识库证据（每条含编号 kb_id、问题、若干根因与方案）③系统草拟答案。
你的任务：审查草拟答案是否覆盖了证据中**所有与用户问题相关**的根因。
- 对每一个证据中存在、与用户问题确实相关、但草拟答案**尚未覆盖**的根因，
  写一条简短补充句（根因+方案+「（依据 kb_id）」）。
- 若草拟答案已覆盖证据中全部相关根因，或证据中没有任何可补充的相关根因，则输出空列表 add=[]。
- 严禁补充证据之外的根因或方案；只补证据里已有、且确实相关的根因。
只输出 JSON：{"add":[{"kb_id":"KBxxx","cause":"...","solution":"...","sentence":"补充句（依据 KBxxx）"}]}"""

BACKFILL_USER = """用户问题：
{query}

知识库证据（共 {n} 条）：
{evidence}

系统草拟答案：
{answer}

请审查草拟答案是否覆盖证据中全部相关根因，输出 JSON。"""


def format_evidence(kb_ids, kb_by_id):
    lines = []
    for i, kid in enumerate(kb_ids, 1):
        e = kb_by_id.get(kid)
        if not e:
            continue
        lines.append(f"[{i}] ({kid}) 问题: {e['problem']}")
        for c in e["causes"]:
            lines.append(f"  根因: {c['cause']} → 方案: {c.get('solution','')}")
    return "\n".join(lines)


def backfill(query, answer, kb_ids, kb_by_id):
    """Call2：返回追加到答案末尾的补充句列表；空列表=无需补。"""
    evidence = format_evidence(kb_ids, kb_by_id)
    if not evidence.strip():
        return []
    user_msg = BACKFILL_USER.format(query=query, n=len(kb_ids),
                                    evidence=evidence, answer=answer)
    content = call_model(BACKFILL_MODEL, [
        {"role": "system", "content": BACKFILL_SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    try:
        data = _extract_json(content)
        add = data.get("add", [])
        if isinstance(add, list):
            return [a for a in add if isinstance(a, dict) and a.get("sentence")]
    except Exception:
        pass
    return []


def summarize(records, label):
    rs = [r for r in records if r.get("in_scope") and not r.get("rejected", True)]
    sc = Counter(r.get(label) for r in rs)
    lc = Counter(r.get("lenient") for r in rs)
    n_in = sum(1 for r in records if r.get("in_scope"))
    by_cat = defaultdict(Counter)
    for r in rs:
        by_cat[r.get("category", "?")][r.get(label)] += 1
    return {
        "in_scope": n_in,
        "answered": len(rs),
        label: dict(sc),
        "strict_task_success": round(sc["correct"] / n_in, 4) if n_in else 0,
        "lenient": dict(lc),
        "lenient_usable_rate": round(lc["usable"] / n_in, 4) if n_in else 0,
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }


def main(limit=None):
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")

    eval_set = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
    if limit:
        eval_set = eval_set[:limit]
    kb_by_id = load_kb()
    gold_by = {e["id"]: e["gold"] for e in eval_set}

    # Call1 基线 = B2 + 一遍法（复用 onepass 的 agent）
    kb_b2 = build_kb_chunks("b2")
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever = make_kb_retriever(kb_b2, bm25_b2, vec_b2)
    skill = DiagnoseSkill(retriever=retriever, expand_parent=False)
    agent = GraphDiagnoseAgent(skill=skill)
    print(f"[init] B2 块数={len(kb_b2)}；一遍法；共 {len(eval_set)} 题")

    strict_cache = load_cache(CACHE_PATH)
    lenient_cache = load_cache(LENIENT_PATH)
    backfill_cache = load_cache(BACKFILL_CACHE)

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
            # ---- Call1 基线生成 + 判官 ----
            out = agent.answer(q)
            rec["rejected"] = out["rejected"]
            rec["kb_ids"] = out["kb_ids"]
            rec["baseline_answer"] = out["answer"]
            if in_scope and out["rejected"]:
                rec["baseline_strict"] = rec["lenient"] = "missed"
            elif not in_scope and not out["rejected"]:
                rec["baseline_strict"] = rec["lenient"] = "false_answer"
            elif not in_scope:
                rec["baseline_strict"] = rec["lenient"] = "ok_reject"
            else:
                sv, lv = judge_case(q, gold, out["answer"], strict_cache, lenient_cache, kb_by_id)
                rec["baseline_strict"], rec["lenient"] = sv, lv

            # ---- Call2 回填（仅 gold 在上下文且非 correct） ----
            rec["backfilled"] = False
            rec["strict"] = rec["baseline_strict"]
            if (in_scope and not out["rejected"] and rec["baseline_strict"] in ("partial", "wrong")
                    and set(gold).issubset(set(out["kb_ids"]))):
                bkey = hashlib.md5(f"{q}::{out['answer'][:200]}".encode("utf-8")).hexdigest()
                if bkey in backfill_cache:
                    adds = backfill_cache[bkey]
                else:
                    adds = backfill(q, out["answer"], out["kb_ids"], kb_by_id)
                    backfill_cache[bkey] = adds
                    save_cache(BACKFILL_CACHE, backfill_cache)
                if adds:
                    supp = "\n\n补充（完整性复核）：\n" + "\n".join(a["sentence"] for a in adds)
                    new_answer = out["answer"] + supp
                    rec["backfilled"] = True
                    rec["backfill_added"] = [a.get("kb_id") for a in adds]
                    rec["treatment_answer"] = new_answer
                    # 重新判官 treatment
                    tsv, tlv = judge_case(q, gold, new_answer, strict_cache, lenient_cache, kb_by_id)
                    rec["strict"] = tsv
                    rec["lenient"] = tlv
                    # 记录相对于基线的变动
                    rec["delta"] = (tsv != rec["baseline_strict"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(eval_set)}] {item['id']} ERROR {rec['error']}")
        records.append(rec)
        RESULT_PATH.write_text(json.dumps(
            {"summary": None, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_cache(CACHE_PATH, strict_cache)
        save_cache(LENIENT_PATH, lenient_cache)
        if "error" not in rec:
            print(f"[{i}/{len(eval_set)}] {item['id']} base={rec.get('baseline_strict')}"
                  f" -> treat={rec.get('strict')} backfilled={rec.get('backfilled')}")

    base_summary = summarize(
        [{"id": r["id"], "category": r["category"], "in_scope": r["in_scope"],
          "rejected": r.get("rejected", True), "strict": r.get("baseline_strict"),
          "lenient": r.get("lenient")} for r in records], "strict")
    treat_summary = summarize(records, "strict")
    RESULT_PATH.write_text(json.dumps(
        {"baseline_summary": base_summary, "treatment_summary": treat_summary, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== 回填实验基线 vs 处理 ==========")
    print("基线 strict =", base_summary["strict_task_success"], dict(base_summary["strict"]))
    print("处理 strict =", treat_summary["strict_task_success"], dict(treat_summary["strict"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
