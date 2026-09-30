"""KB016 知识库补全后的失真消除验证（单题 eval_080）。

目的：补 KB016 根因前，eval_080 因 KB016.causes=[] 使 build_reference 生成空参照，
严格判官按「覆盖全部根因」判 → trivially correct（失真）。补 3 条真实根因后：
  1) 主路 B2 索引已重建，KB016 块含真实根因；
  2) 本脚本用**独立新缓存文件**强制重判 eval_080，证明失真消除、判分基于真实根因；
  3) 打印模型是否真能基于 KB016 新根因作答（真实能力，不再 trivially correct）。
"""
import os
os.environ["TWO_PASS"] = "0"  # 确保一遍法（与 C 组最优组合一致）
import sys
import json
import time
import hashlib
from pathlib import Path

SRC = Path("D:/3D打印故障诊断Agent/src")
sys.path.insert(0, str(SRC))
ROOT = Path("D:/3D打印故障诊断Agent")

from config import KB_V2_PATH, EVAL_SET_PATH, DASHSCOPE_API_KEY
from agent_graph import GraphDiagnoseAgent

JUDGE_MODEL = "qwen-turbo"
STRICT_SYSTEM = """你是 RAG 系统答案质量评审员。给定用户问题、标准参照
（gold 知识库条目）和系统答案，判断答案是否正确。
- correct：答案覆盖参照中**全部**主要根因，方案与参照一致。
- partial：只覆盖部分主要根因，或方案不完整。
- wrong：基本未覆盖 / 与参照矛盾 / 答非所问。
只输出 JSON：{"verdict": "correct|partial|wrong", "reason": "一句话"}"""
USER_TMPL = """用户问题：
{query}

标准参照（gold 知识库条目）：
{reference}

系统答案：
{answer}

请输出 JSON 评审结果。"""

SOLUTION_PREVIEW_CHARS = 250
VERIFY_CACHE = ROOT / "exp" / "_kb016_verify_cache.json"  # 独立缓存，强制重判


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


def call_judge(system, user_msg, retries=3):
    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(model=JUDGE_MODEL, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}],
                result_format="message", temperature=0.0)
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                m = __import__("re").search(r"\{.*\}", content, __import__("re").S)
                if m:
                    return json.loads(m.group(0))
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"judge 失败: {last_err}")


def main():
    kb_by_id = load_kb()
    e80 = next(x for x in json.load(open(EVAL_SET_PATH, encoding="utf-8")) if x["id"] == "eval_080")
    print(f"=== eval_080 ===\nquery: {e80['query']}\ncategory: {e80['category']} | gold: {e80['gold']}\n")

    # 1) 主路生成
    agent = GraphDiagnoseAgent()
    out = agent.answer(e80["query"])
    print(f"rejected: {out['rejected']} | 检索 kb_ids(top-5): {out['kb_ids']}\n")
    answer = out["answer"]
    print(f"--- 系统答案 ---\n{answer}\n")

    # 2) 真实参照（含 KB016 新根因）
    ref = build_reference(e80["gold"], kb_by_id)
    print(f"--- 标准参照（真实根因）---\n{ref}\n")

    # 3) 强制重判（独立缓存）
    cache = json.loads(VERIFY_CACHE.read_text(encoding="utf-8")) if VERIFY_CACHE.exists() else {}
    skey = hashlib.md5(f"v1::{e80['query']}::{answer[:200]}".encode()).hexdigest()
    if skey in cache:
        j = cache[skey]
    else:
        j = call_judge(STRICT_SYSTEM, USER_TMPL.format(query=e80["query"], reference=ref, answer=answer))
        cache[skey] = j
        VERIFY_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 严格判官 verdict: {j.get('verdict')} ===")
    print(f"reason: {j.get('reason')}")

    # 4) 失真前后对照
    n_causes = len(kb_by_id["KB016"]["causes"])
    print(f"\n[失真消除确认] KB016 当前根因数 = {n_causes}（补前=0）")
    print(f"  补前: 空参照 → trivially correct（失真）")
    print(f"  补后: 参照含 {n_causes} 条真实根因 → 判官按真实覆盖判分（verdict={j.get('verdict')}）")


if __name__ == "__main__":
    main()
