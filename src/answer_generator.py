"""P2 答案生成器：把 Skill 返回的证据 + 用户问题 → 结构化诊断答案。

设计要点
--------
1. **上下文管理（harness 职责）**：证据全文可能很长（parent-child 扩展后
   一个 problem 最多 18 个 cause），生成前按预算裁剪——
   检索命中块优先保全文，扩展块只保摘要，总量不超过 MAX_CONTEXT_CHARS。
2. **严格接地（grounded）**：prompt 明确「只用给定证据、禁止编造、
   证据不足就说不足」，与拒答层形成双保险。
3. **磁盘缓存**：key 含 PROMPT_VERSION / model / top_k / expand 标志，
   改 prompt 必须递增版本号，否则命中旧缓存（同 rejector 约定）。
"""
import hashlib
import json
import re
import time
from pathlib import Path

from config import DASHSCOPE_API_KEY, PROJECT_ROOT

ANSWER_MODEL = "qwen-turbo"
PROMPT_VERSION = "v1"           # 改 prompt 时必须递增
CACHE_PATH = PROJECT_ROOT / "data" / "answer_cache.json"
MAX_CONTEXT_CHARS = 8000        # 喂给生成模型的证据总预算
EXPANDED_CHUNK_CHARS = 400      # 扩展块（非直接命中）每块上限

SYSTEM_PROMPT = """你是 FDM 3D 打印故障诊断专家。用户描述了一个打印问题，
系统已从排障知识库中检索到相关条目（含根因与解决方案）。

请严格基于给定证据作答：
1. 只用证据中的信息，**禁止编造**证据外的参数、步骤或结论。
2. 按「最可能的根因 → 对应解决方案」组织答案，多个根因按证据顺序列出。
3. 给出具体可操作的调整项（参数值、检查步骤），保留证据中的关键数值。
4. 用与用户提问相同的语言回答。
5. 若证据与问题明显不相关或不足以诊断，直接说明证据不足，不要硬答。

只输出 JSON，不要任何其他内容：
{"answer": "你的完整诊断回答"}"""

USER_TEMPLATE = """用户问题：
{query}

知识库证据（{n} 条，retrieved=直接命中，expanded=同问题下的补充根因）：
{context}

请输出 JSON 格式的诊断回答。"""


def build_context(evidence, max_chars=MAX_CONTEXT_CHARS,
                  expanded_chars=EXPANDED_CHUNK_CHARS):
    """上下文预算裁剪：检索命中块优先全文，扩展块摘要，超预算截断。"""
    lines, used = [], 0
    for i, e in enumerate(evidence, 1):
        tag = "expanded" if e.get("expanded") else "retrieved"
        text = e["text"]
        if e.get("expanded") and len(text) > expanded_chars:
            text = text[:expanded_chars] + " …[截断]"
        block = (f"[{i}] ({e['kb_id']}, {tag}) 问题: {e['problem']}\n{text}")
        if used + len(block) > max_chars:
            lines.append("…[上下文预算用尽，其余证据略]")
            break
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines)


def _extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def _cache_key(query, model, top_k, expand, ev_fp):
    return hashlib.md5(
        f"{PROMPT_VERSION}::{model}::top{top_k}::exp{int(expand)}"
        f"::ev{ev_fp}::{query}".encode("utf-8")).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


class AnswerGenerator:
    def __init__(self, model=ANSWER_MODEL):
        self.model = model

    def generate(self, query, evidence, top_k=5, expand=True,
                 temperature=0.0, retries=3):
        """生成诊断答案（带磁盘缓存）。返回 {answer, raw, cached}。

        ⚠️ 缓存 key 含证据指纹（证据内全部 chunk_id 的排序散列）。
        证据构成变了（如 parent-child 扩展策略调整）→ key 失效 → 自动重新生成，
        避免命中旧证据生成的过期答案。
        """
        cache = load_cache()
        ev_fp = hashlib.md5(
            "|".join(sorted(e["chunk_id"] for e in evidence)).encode("utf-8")
        ).hexdigest()[:8]
        ck = _cache_key(query, self.model, top_k, expand, ev_fp)
        if ck in cache:
            return {**cache[ck], "cached": True}

        import dashscope
        from dashscope import Generation
        key = DASHSCOPE_API_KEY
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY not set，无法生成答案")
        dashscope.api_key = key

        user_msg = USER_TEMPLATE.format(
            query=query, n=len(evidence), context=build_context(evidence))
        last_err = None
        for attempt in range(retries):
            try:
                resp = Generation.call(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    result_format="message",
                    temperature=temperature,
                )
                if resp.status_code == 200:
                    content = resp.output.choices[0].message.content
                    try:
                        data = _extract_json(content)
                        answer = str(data.get("answer", content))
                    except (json.JSONDecodeError, AttributeError):
                        answer = content  # 模型没按 JSON 输出时兜底取原文
                    entry = {"answer": answer, "raw": content[:500]}
                    cache[ck] = entry
                    save_cache(cache)
                    return {**entry, "cached": False}
                last_err = f"{resp.code}: {resp.message}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"答案生成调用失败: {last_err}")
