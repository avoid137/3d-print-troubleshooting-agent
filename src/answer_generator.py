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

ANSWER_MODEL = "qwen-plus-2025-07-28"   # 最佳配置：qwen-max 实测 35.00% < qwen-plus 39.37%（过生成触发严格判官惩罚）
PROMPT_VERSION = "v2.1"         # 改 prompt / 上下文裁剪时必须递增（缓存 key 含此版本）
CACHE_PATH = PROJECT_ROOT / "data" / "answer_cache.json"
MAX_CONTEXT_CHARS = 10000       # 喂给生成模型的证据总预算（v2 上调，容纳更全的根因）
EXPANDED_CHUNK_CHARS = 700      # 扩展块（非直接命中）每块上限（v2 上调，避免 gold 根因被截断）

SYSTEM_PROMPT = """你是 FDM 3D 打印故障诊断专家。系统已从排障知识库检索到若干「证据」条目，
每条含编号(kb_id)、问题(problem)、若干根因(cause)与对应解决方案(solution)。
你必须**严格只基于这些证据**作答，不得依赖自身知识。

硬性规则：
1. 相关才穷尽：对与用户问题**相关**的每条证据，必须列出其中**全部**根因及对应方案；
   禁止用概括性语言跳过任何一条相关根因。若某证据块明显与问题无关，
   可跳过并注明「该证据与问题关联性弱，未采纳」，不要罗列其根因。
2. 仅用证据：答案里的每条根因/方案都必须能在上述证据中找到对应内容；
   **严禁**补充证据之外的根因或对策（这类内容会被判为错误）。
3. 标注出处：每条根因/方案后用「（依据 {kb_id}）」注明，kb_id 必须来自
   **上方真实证据块**中实际出现的编号；禁止引用示例、杜撰或证据中不存在的编号。
4. 排序：同一证据内多条根因按证据原有顺序；多条证据按编号顺序。
5. 语言：与用户提问相同语言。
6. 证据不足：若所有证据都与问题明显无关，直接说明「知识库未检索到相关条目」，不要硬答。

只输出 JSON，不要任何其他内容：
{"answer": "你的完整诊断回答"}"""

USER_TEMPLATE = """用户问题：
{query}

知识库证据（共 {n} 条，retrieved=直接命中，expanded=同问题下的补充根因；
请**逐条处理，覆盖每条证据中的全部根因**，不要遗漏）：
{context}

请输出 JSON 格式的诊断回答。"""

# 通用 few-shot（不取自评测集，避免过拟合）：演示「穷尽全部根因 + 标注出处 + 不补外部知识」
# 注意：示例用合成编号 KB-Demo，且规则 3 已禁止引用示例/杜撰编号，避免模型把示例 id 照搬进真实答案。
FEWSHOT_USER = """用户问题：
打印件的第一层总是翘边、粘不牢，轻轻一碰就脱落。

知识库证据（共 1 条，retrieved=直接命中）：
[1] (KB-Demo, retrieved) 问题: 第一层附着不良
根因: 打印床未调平，喷嘴与床面间隙不均 → 方案: 重新调平打印床，用一张 A4 纸测试间隙。
根因: 热床温度过低 → 方案: 将热床温度提高到 60°C 左右（PLA）。
根因: 床面有油脂或碎屑 → 方案: 用酒精擦拭床面，必要时涂固体胶棒增加附着力。

请输出 JSON 格式的诊断回答。"""

FEWSHOT_ASSISTANT = """{"answer": "第一层翘边脱落的可能根因与对策：\\n1. 打印床未调平、喷嘴与床面间隙不均：重新调平打印床，用一张 A4 纸测试间隙是否均匀（依据 KB-Demo）。\\n2. 热床温度过低：将热床温度提高到约 60°C（PLA 适用）（依据 KB-Demo）。\\n3. 床面有油脂或碎屑：用酒精擦拭床面，必要时涂固体胶棒增加附着力（依据 KB-Demo）。"}"""


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
                        {"role": "user", "content": FEWSHOT_USER},
                        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
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
