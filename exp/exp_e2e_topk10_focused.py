"""定向实验：top-k=10 的上行实测（仅 16 道「k=5 缺口、k=10 才进上下文」的题）。

不做全量 200（避免 ~90min + 上下文稀释风险盲跑），只测这 16 题在 k=10 下
能否借「gold 进上下文」转 correct。下行（k=10 对原 125 覆盖题的稀释）另议。

对比基线 = exp_e2e_kb_backfill_prod_result.json 中该 id 的 k=5+treatment  verdict。
"""
import os
os.environ["TWO_PASS"] = "0"
os.environ["ANSWER_BACKFILL"] = "1"

import sys
import json
import hashlib
from collections import Counter

sys.path.insert(0, "src")
ROOT = "D:/3D打印故障诊断Agent"

import faiss  # noqa: E402
from config import (KB_V2_PATH, EVAL_SET_PATH, EMBED_DIM, EMBED_MODEL,  # noqa: E402
                    DASHSCOPE_API_KEY)  # noqa: E402
from bm25_index import tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from vector_index import _dashscope_embed  # noqa: E402
from skill_diagnose import DiagnoseSkill  # noqa: E402
from agent_graph import GraphDiagnoseAgent  # noqa: E402

JUDGE_MODEL = "qwen-turbo"
STRICT_VERSION = "v1"
SOLUTION_PREVIEW_CHARS = 250
CACHE_PATH = f"{ROOT}/data/judge_e2e_cache.json"


def build_kb_chunks():
    kb = json.load(open(KB_V2_PATH, encoding="utf-8"))
    chunks = []
    for it in kb:
        causes = it.get("causes") or []
        body = "\n".join(f"{i+1}. {c['cause']}: {c.get('solution', '')}"
                         for i, c in enumerate(causes))
        chunks.append({"chunk_id": it["kb_id"], "kb_id": it["kb_id"],
                       "problem": it["problem"], "type": "kb",
                       "text": it["problem"] + ("\n" + body if body else "")})
    return chunks


def build_kb_index(chunks):
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    vecs = _dashscope_embed([c["text"] for c in chunks])
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return bm25, idx


def load_kb():
    return {e["kb_id"]: e for e in json.load(open(KB_V2_PATH, encoding="utf-8"))}


def build_reference(gold, kb_by_id):
    blocks = []
    for kb_id in gold:
        e = kb_by_id.get(kb_id)
        if not e:
            continue
        lines = [f"【{kb_id}】问题: {e['problem']}"]
        for c in e["causes"]:
            lines.append(f"  根因: {c['cause']}\n  方案: {c['solution'][:SOLUTION_PREVIEW_CHARS]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _extract_json(text):
    import re
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def judge(query, gold, answer, strict_cache, kb_by_id):
    ref = build_reference(gold, kb_by_id)
    sys_mod = __import__("dashscope")
    from dashscope import Generation
    sys_mod.api_key = DASHSCOPE_API_KEY
    skey = hashlib.md5(f"{STRICT_VERSION}::{query}::{answer[:200]}".encode()).hexdigest()
    if skey in strict_cache:
        return strict_cache[skey]["verdict"]
    resp = Generation.call(model=JUDGE_MODEL, messages=[
        {"role": "system", "content": "你是 RAG 系统答案质量评审员。给定用户问题、标准参照（gold 知识库条目）和系统答案，判断答案是否正确。\n- correct：答案覆盖参照中**全部**主要根因，方案与参照一致。\n- partial：只覆盖部分主要根因，或方案不完整。\n- wrong：基本未覆盖 / 与参照矛盾 / 答非所问。\n只输出 JSON：{\"verdict\": \"correct|partial|wrong\", \"reason\": \"一句话\"}"},
        {"role": "user", "content": f"用户问题：\n{query}\n\n标准参照（gold 知识库条目）：\n{ref}\n\n系统答案：\n{answer}\n\n请输出 JSON 评审结果。"}],
        result_format="message", temperature=0.0)
    v = _extract_json(resp.output.choices[0].message.content).get("verdict", "wrong")
    if v not in ("correct", "partial", "wrong"):
        v = "wrong"
    strict_cache[skey] = {"verdict": v}
    return v


def main():
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    ev = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
    by_id = {e["id"]: e for e in ev}
    kb_by_id = load_kb()

    # 16 道 k=10 才进上下文的题（来自 exp_topk10_analysis）
    prod = json.load(open(f"{ROOT}/exp/exp_e2e_kb_backfill_prod_result.json", encoding="utf-8"))
    prod_by = {r["id"]: r for r in prod["records"]}
    recovered = ['eval_010', 'eval_022', 'eval_028', 'eval_032', 'eval_034', 'eval_044',
                 'eval_070', 'eval_092', 'eval_096', 'eval_102', 'eval_103', 'eval_116',
                 'eval_128', 'eval_145', 'eval_148', 'eval_175']

    kb_b2 = build_kb_chunks()
    bm25_b2, vec_b2 = build_kb_index(kb_b2)
    retriever = HybridRetriever(vector_backend="faiss")
    retriever.chunks = kb_b2
    retriever.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(kb_b2)}
    retriever.bm25 = bm25_b2
    retriever.vector_index = vec_b2
    retriever.vector_meta = {"method": "dashscope", "model": EMBED_MODEL, "dim": EMBED_DIM}
    retriever.embed_method = "dashscope"
    skill = DiagnoseSkill(retriever=retriever, expand_parent=False)
    agent = GraphDiagnoseAgent(skill=skill)

    strict_cache = json.load(open(CACHE_PATH, encoding="utf-8")) if __import__("os").path.exists(CACHE_PATH) else {}

    rows = []
    for iid in recovered:
        e = by_id[iid]
        q, gold = e["query"], e["gold"]
        out = agent.answer(q, top_k=10)   # k=10 检索 + 生成 + 回填
        v10 = judge(q, gold, out["answer"], strict_cache, kb_by_id)
        v5 = prod_by[iid].get("strict")    # k=5 + 回填 基线
        rows.append({"id": iid, "category": e["category"], "k5": v5, "k10": v10,
                     "kb_ids10": out["kb_ids"], "backfilled": out.get("trace", [{}])[-1].get("backfilled", False)})
        json.dump({"rows": rows}, open(f"{ROOT}/exp/exp_e2e_topk10_focused_result.json", "w"), ensure_ascii=False, indent=2)

    c5 = Counter(r["k5"] for r in rows)
    c10 = Counter(r["k10"] for r in rows)
    improved = [r["id"] for r in rows if r["k5"] != "correct" and r["k10"] == "correct"]
    worsened = [r["id"] for r in rows if r["k5"] == "correct" and r["k10"] != "correct"]
    print(f"定向 16 题（k=5+treatment 基线 -> k=10+treatment）:")
    print(f"  k5 verdict: {dict(c5)}")
    print(f"  k10 verdict: {dict(c10)}")
    print(f"  转 correct (上行): {len(improved)} {improved}")
    print(f"  退化: {len(worsened)} {worsened}")
    print(f"  净增 correct: {c10['correct'] - c5['correct']}")


if __name__ == "__main__":
    main()
