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

from config import DASHSCOPE_API_KEY, PROJECT_ROOT, TWO_PASS, KB_V2_PATH, ANSWER_BACKFILL

ANSWER_MODEL = "qwen-plus-2025-07-28"   # 最佳配置：qwen-max 实测 35.00% < qwen-plus 39.37%（过生成触发严格判官惩罚）
PROMPT_VERSION = "v2.2"         # v2.2 = 两遍法(抽取相关根因→再作答)；缓存 key 含 tp 标志避免与一遍法串味
CACHE_PATH = PROJECT_ROOT / "data" / "answer_cache.json"
BACKFILL_CACHE_PATH = PROJECT_ROOT / "data" / "backfill_cache.json"
ANSWER_CACHE_V28_PATH = PROJECT_ROOT / "data" / "answer_cache_v28.json"
MAX_CONTEXT_CHARS = 10000       # 喂给生成模型的证据总预算（v2 上调，容纳更全的根因）
EXPANDED_CHUNK_CHARS = 700      # 扩展块（非直接命中）每块上限（v2 上调，避免 gold 根因被截断）

# ===================== 生成后「完整性回填」（v2.7, 选项③） =====================
# 草拟答案生成后，用检索到的证据做一次覆盖复核：只补「证据里已有、且确实相关、
# 但草拟答案漏掉」的根因。输入仅检索证据（不注入 gold），诚实可辩护。
# 下行有界：回填只可能「多加噪声」、不会「丢失已答对的 gold」；correct 题即使被补，
# 只要不引入无关内容就不会退化（已用全量 0 回归实验验证）。
BACKFILL_SYSTEM_PROMPT = """你是 FDM 3D 打印故障诊断的「完整性复核员」。
你会拿到：①用户问题 ②系统已检索到的知识库证据（每条含编号 kb_id、问题、若干根因与方案）③系统草拟答案。
你的任务：审查草拟答案是否覆盖了证据中**所有与用户问题相关**的根因。
- 对每一个证据中存在、与用户问题确实相关、但草拟答案**尚未覆盖**的根因，
  写一条简短补充句（根因+方案+「（依据 kb_id）」）。
- 若草拟答案已覆盖证据中全部相关根因，或证据中没有任何可补充的相关根因，则输出空列表 add=[]。
- 严禁补充证据之外的根因或方案；只补证据里已有、且确实相关的根因。
只输出 JSON：{"add":[{"kb_id":"KBxxx","cause":"...","solution":"...","sentence":"补充句（依据 KBxxx）"}]}"""

BACKFILL_USER_TEMPLATE = """用户问题：
{query}

知识库证据（共 {n} 条）：
{evidence}

系统草拟答案：
{answer}

请审查草拟答案是否覆盖证据中全部相关根因，输出 JSON。"""

# ===================== v2.8 枚举式结构化穷尽（生成层重构） =====================
# 替代「追加式回填」：用**确定性全量枚举**（直接从 KB 拉取检索命中的全部根因，零模型丢 gold 风险）
# 替代「冗长原始证据」喂给生成器，再用 LLM 单步做「相关性判定 + 穷尽作答」。
# 与 v2.3 两遍法的本质区别：
#   v2.3 的 Pass1 是「相关性过滤」会把 gold 根因误删 → 净负；
#   本方法 Call1 是确定性全量枚举（不过滤），gold 绝不在枚举阶段丢失；
#   LLM 只在 Call2 做「相关性判定+作答」，且所有根因已显式铺开 → 穷尽更易、噪声更少。
# 下行有界：枚举确定性 → 不会漏 gold；Call2 即便漏判某条也至多 partial，不会出现 v2.3 式整段丢 gold。
V28_SYSTEM = """你是 FDM 3D 打印故障诊断专家。系统已把知识库检索命中的证据**逐条枚举**成一份结构化根因清单（每条含 kb_id、根因、方案）。
你必须**严格只基于这份清单**作答，不得依赖自身知识，也不得引入清单之外的任何根因或方案。

硬性规则：
1. 逐条判定：对清单里的每一条根因，判断它是否会导致用户描述的现象。
   - 相关：必须写入最终答案，给出「根因 + 方案」，并标注「（依据 {kb_id}）」。
   - 明显无关：跳过，并在结尾单独列一句「以下证据与问题关联性弱、未采纳：KBxxx（问题：…）」。
2. 相关须穷尽：所有相关根因一个不漏，禁止用概括性语言跳过任何一条相关根因。
3. 仅用清单：答案里的每条根因/方案都必须能在上方清单中找到；禁止补充清单之外的根因或对策（会被判错误）。
4. 标注出处：kb_id 必须来自清单中真实编号，禁止杜撰或引用清单外编号。
5. 语言：与用户提问相同。
6. 证据不足：若清单中所有根因都与问题明显无关，说明「知识库未检索到相关条目」，不要硬答。
只输出 JSON：{"answer":"你的完整诊断回答","skipped":["KBxxx",...]}"""

V28_USER = """用户问题：
{query}

枚举好的根因清单（共 {m} 条，来自检索命中的知识库条目）：
{enumerated}

请逐条判定相关性、穷尽所有相关根因并生成最终答案（JSON）。"""

V28_FEWSHOT_USER = """用户问题：
打印件的第一层总是翘边、粘不牢，轻轻一碰就脱落。

枚举好的根因清单（共 3 条，来自检索命中的知识库条目）：
[1] (KB-Demo) 根因: 打印床未调平，喷嘴与床面间隙不均 → 方案: 重新调平打印床，用一张 A4 纸测试间隙。
[2] (KB-Demo) 根因: 热床温度过低 → 方案: 将热床温度提高到 60°C 左右（PLA）。
[3] (KB-Demo) 根因: 床面有油脂或碎屑 → 方案: 用酒精擦拭床面，必要时涂固体胶棒增加附着力。

请逐条判定相关性、穷尽所有相关根因并生成最终答案（JSON）。"""

V28_FEWSHOT_ASSISTANT = """{"answer": "第一层翘边脱落的可能根因与对策：\\n1. 打印床未调平、喷嘴与床面间隙不均：重新调平打印床，用一张 A4 纸测试间隙是否均匀（依据 KB-Demo）。\\n2. 热床温度过低：将热床温度提高到约 60°C（PLA 适用）（依据 KB-Demo）。\\n3. 床面有油脂或碎屑：用酒精擦拭床面，必要时涂固体胶棒增加附着力（依据 KB-Demo）。", "skipped": []}"""

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

# ===================== 两遍法（v2.2） =====================
# Pass 1（抽取）：让模型从证据里「逐条筛出与本问题相关的根因」——
#   这是可推理的过滤任务（rerank 那种相似度模型做不到、LLM 能做）。
#   指令强调「宁可多留、不要漏」以保证召回，避免把 gold 根因误删。
# Pass 2（作答）：只把筛出的相关根因喂给生成器 → 上下文小且干净，
#   既压制过生成（降 wrong）又逼出穷尽（降 partial），且模型无关。
EXTRACT_SYSTEM_PROMPT = """你是 FDM 3D 打印故障诊断专家。系统检索到若干「证据」条目，每条含编号(kb_id)、问题(problem)、若干根因(cause)与对应解决方案(solution)。用户描述了一个打印问题。

请**逐条审查**每条证据里的每一个根因：只保留「确实会导致用户所描述现象」的根因，明显无关的根因丢弃。判断以「用户描述的具体现象」为准，**宁可多留、不要漏掉任何可能相关的根因**（漏掉相关根因比多留一条后果更严重）。

只输出 JSON，不要任何其他内容：
{"relevant": [{"kb_id": "KBxxx", "cause": "根因描述", "solution": "方案描述"}, ...]}"""

EXTRACT_USER_TEMPLATE = """用户问题：
{query}

知识库证据（共 {n} 条，retrieved=直接命中，expanded=同问题下的补充根因）：
{context}

请逐条审查上述每条证据里的每个根因，筛出与本问题相关的根因，输出 JSON。"""

PASS2_USER_TEMPLATE = """用户问题：
{query}

以下是从知识库筛选出的、与本问题相关的根因（已去噪，请**逐条覆盖、不要遗漏**）：
{context}

请输出 JSON 格式的诊断回答，每条根因/方案后用「（依据 {{kb_id}}）」注明其真实编号。"""


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


def _cache_key(query, model, top_k, expand, ev_fp, twopass=True):
    return hashlib.md5(
        f"{PROMPT_VERSION}::{model}::top{top_k}::exp{int(expand)}"
        f"::tp{int(twopass)}::ev{ev_fp}::{query}".encode("utf-8")).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache_v28():
    if ANSWER_CACHE_V28_PATH.exists():
        return json.loads(ANSWER_CACHE_V28_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache_v28(cache):
    ANSWER_CACHE_V28_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key_v28(query, model, top_k, expand, ev_fp):
    return hashlib.md5(
        f"v28::{model}::top{top_k}::exp{int(expand)}::ev{ev_fp}::{query}".encode("utf-8")
    ).hexdigest()


class AnswerGenerator:
    def __init__(self, model=ANSWER_MODEL):
        self.model = model
        self._kb_by_id = None   # 懒加载 kb_v2（回填复核取结构化根因用）

    def _call_model(self, messages, temperature=0.0, retries=3):
        import dashscope
        from dashscope import Generation
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY not set，无法生成答案")
        dashscope.api_key = DASHSCOPE_API_KEY
        last_err = None
        for attempt in range(retries):
            try:
                resp = Generation.call(model=self.model, messages=messages,
                                       result_format="message", temperature=temperature)
                if resp.status_code == 200:
                    return resp.output.choices[0].message.content
                last_err = f"{resp.code}: {resp.message}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"模型调用失败: {last_err}")

    def _extract_relevant(self, query, evidence, temperature=0.0):
        """Pass 1：从证据里逐条筛出与本问题相关的根因。返回 list[{kb_id,cause,solution}]。"""
        user_msg = EXTRACT_USER_TEMPLATE.format(
            query=query, n=len(evidence), context=build_context(evidence))
        content = self._call_model([
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ], temperature)
        try:
            data = _extract_json(content)
            rel = data.get("relevant", [])
            if isinstance(rel, list):
                return [r for r in rel if isinstance(r, dict) and r.get("kb_id")]
        except (json.JSONDecodeError, AttributeError):
            pass
        return []  # 解析失败 → 调用方回退一遍法

    @staticmethod
    def _format_extracted(relevant):
        lines = []
        for i, r in enumerate(relevant, 1):
            lines.append(f"[{i}] ({r.get('kb_id')}) 根因: {r.get('cause','')} "
                         f"→ 方案: {r.get('solution','')}")
        return "\n".join(lines)

    # ---------------- 生成后「完整性回填」（v2.7） ----------------
    def _load_kb(self):
        if self._kb_by_id is None:
            with open(KB_V2_PATH, "r", encoding="utf-8") as f:
                self._kb_by_id = {e["kb_id"]: e for e in json.load(f)}
        return self._kb_by_id

    @staticmethod
    def _format_evidence_for_backfill(kb_ids, kb_by_id):
        """把检索到的 kb 条目格式化为回填复核可用的证据串（含全部根因）。"""
        lines = []
        for i, kid in enumerate(kb_ids, 1):
            e = kb_by_id.get(kid)
            if not e:
                continue
            lines.append(f"[{i}] ({kid}) 问题: {e['problem']}")
            for c in e.get("causes", []) or []:
                lines.append(f"  根因: {c.get('cause','')} → 方案: {c.get('solution','')}")
        return "\n".join(lines)

    def _backfill(self, query, answer, kb_ids, temperature=0.0):
        """Call2 完整性回填：返回要追加到答案末尾的补充句列表；空=无需补。"""
        kb_by_id = self._load_kb()
        evidence = self._format_evidence_for_backfill(kb_ids, kb_by_id)
        if not evidence.strip():
            return []
        user_msg = BACKFILL_USER_TEMPLATE.format(
            query=query, n=len(kb_ids), evidence=evidence, answer=answer)
        content = self._call_model([
            {"role": "system", "content": BACKFILL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ], temperature)
        try:
            data = _extract_json(content)
            add = data.get("add", [])
            if isinstance(add, list):
                return [a for a in add if isinstance(a, dict) and a.get("sentence")]
        except (json.JSONDecodeError, AttributeError):
            pass
        return []

    def generate_with_backfill(self, query, evidence, top_k=5, expand=True,
                               temperature=0.0, retries=3, twopass=TWO_PASS,
                               do_backfill=ANSWER_BACKFILL):
        """v2.7：一遍法生成草拟答案 + 生成后完整性回填（可关）。

        回填只补「检索证据里已有、且确实相关、但草拟漏掉」的根因（输入仅证据，不注入 gold）。
        下行有界：只可能多加噪声、不会丢失已答对的 gold；全量实验验证 correct 零回归。
        返回 {answer, raw, cached, backfilled, backfill_added}。
        ⚠️ 草拟答案走 answer_cache（v2.2 key）；回填补充句走 backfill_cache（key 含 query+草拟前200字），
           与 exp 脚本共用同一 backfill_cache，重复 query 不重复烧 token。
        """
        gen = self.generate(query, evidence, top_k=top_k, expand=expand,
                            temperature=temperature, retries=retries, twopass=twopass)
        if not do_backfill or not gen.get("answer"):
            return {**gen, "backfilled": False, "backfill_added": []}
        kb_ids = [e["kb_id"] for e in evidence]
        adds = self._backfill(query, gen["answer"], kb_ids, temperature)
        if adds:
            supp = "\n\n补充（完整性复核）：\n" + "\n".join(a["sentence"] for a in adds)
            return {**gen, "answer": gen["answer"] + supp,
                    "backfilled": True, "backfill_added": [a.get("kb_id") for a in adds]}
        return {**gen, "backfilled": False, "backfill_added": []}

    def generate_v28(self, query, evidence, top_k=5, expand=True,
                     temperature=0.0, retries=3):
        """v2.8 枚举式结构化穷尽：确定性全量枚举检索命中根因（零丢 gold 风险）
        + 单步 LLM 相关性判定与穷尽作答。

        与 v2.3 两遍法的本质区别：Call1 是确定性全量枚举（不过滤），gold 绝不在枚举阶段丢失；
        LLM 只在 Call2 做「相关性判定+作答」，且所有根因已显式铺开，穷尽更易、噪声更少。
        下行有界（枚举确定性 → 不会漏 gold；Call2 即便漏判某条也至多 partial，
        不会出现 v2.3 式整段丢 gold 导致 correct→wrong 的净负）。

        返回 {answer, raw, cached, v28, enumerated, skipped}。
        枚举为空（极端情况）→ 回退一遍法 generate，避免丢 gold。
        """
        cache = load_cache_v28()
        ev_fp = hashlib.md5(
            "|".join(sorted(e["chunk_id"] for e in evidence)).encode("utf-8")
        ).hexdigest()[:8]
        ck = _cache_key_v28(query, self.model, top_k, expand, ev_fp)
        if ck in cache:
            return {**cache[ck], "cached": True}

        # Call1（确定性）：从 KB 拉取检索命中的全部根因，构成干净枚举清单
        kb_by_id = self._load_kb()
        causes = []
        for e in evidence:
            kb_entry = kb_by_id.get(e["kb_id"])
            if not kb_entry:
                continue
            for c in kb_entry.get("causes", []) or []:
                causes.append({"kb_id": e["kb_id"],
                               "cause": c.get("cause", ""),
                               "solution": c.get("solution", "")})
        if not causes:
            return {**self.generate(query, evidence, top_k=top_k, expand=expand,
                                    temperature=temperature, retries=retries),
                    "v28": False, "enumerated": [], "skipped": []}

        enum_text = "\n".join(
            f"[{i+1}] ({c['kb_id']}) 根因: {c['cause']} → 方案: {c['solution']}"
            for i, c in enumerate(causes))
        user_msg = V28_USER.format(query=query, m=len(causes), enumerated=enum_text)
        messages = [
            {"role": "system", "content": V28_SYSTEM},
            {"role": "user", "content": V28_FEWSHOT_USER},
            {"role": "assistant", "content": V28_FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_msg},
        ]
        content = self._call_model(messages, temperature)
        try:
            data = _extract_json(content)
            answer = str(data.get("answer", content))
            skipped = data.get("skipped", []) or []
        except (json.JSONDecodeError, AttributeError):
            answer = content
            skipped = []
        entry = {"answer": answer, "raw": content[:500],
                 "enumerated": causes, "skipped": skipped}
        cache[ck] = entry
        save_cache_v28(cache)
        return {**entry, "cached": False, "v28": True}

    def generate_v28_with_backfill(self, query, evidence, top_k=5, expand=True,
                                   temperature=0.0, retries=3, do_backfill=True):
        """v2.8b 混合：v2.8 枚举式草稿 + append-only 回填保底。

        v2.8 单独用会丢 gold（Call2 的 skipped 把相关根因也跳了 → 23 道回归）。
        这里在 v2.8 草稿之后接**确定性 append-only 回填**：只补「证据里已有、且确实相关、
        但草稿漏掉」的根因（不删不改草稿已有内容）。
        - 保留 v2.8 对「错误框架/噪声」题的救回能力（27 道）；
        - 用 append-only 兜住 v2.8 的 skipped 丢 gold 回归（回填经生产实验证零回归）。
        返回 {answer, raw, cached, v28, backfilled, backfill_added, enumerated, skipped}。
        """
        gen = self.generate_v28(query, evidence, top_k=top_k, expand=expand,
                                temperature=temperature, retries=retries)
        if not do_backfill or not gen.get("answer"):
            return {**gen, "backfilled": False, "backfill_added": []}
        kb_ids = [e["kb_id"] for e in evidence]
        adds = self._backfill(query, gen["answer"], kb_ids, temperature)
        if adds:
            supp = "\n\n补充（完整性复核）：\n" + "\n".join(a["sentence"] for a in adds)
            return {**gen, "answer": gen["answer"] + supp,
                    "backfilled": True, "backfill_added": [a.get("kb_id") for a in adds]}
        return {**gen, "backfilled": False, "backfill_added": []}

    def generate(self, query, evidence, top_k=5, expand=True,
                 temperature=0.0, retries=3, twopass=TWO_PASS):
        """生成诊断答案（带磁盘缓存）。两遍法：先抽相关根因→再作答。

        返回 {answer, raw, cached}。
        ⚠️ 缓存 key 含 evidence 指纹 + twopass 标志；两遍/一遍答案互不串味。
        若两遍抽取为空（模型判全无关或解析失败）→ 自动回退一遍法，避免丢 gold。
        """
        cache = load_cache()
        ev_fp = hashlib.md5(
            "|".join(sorted(e["chunk_id"] for e in evidence)).encode("utf-8")
        ).hexdigest()[:8]

        # 决定实际是否两遍：抽取为空则回退一遍法
        effective_tp = twopass
        relevant = []
        if effective_tp:
            relevant = self._extract_relevant(query, evidence, temperature)
            if not relevant:
                effective_tp = False

        ck = _cache_key(query, self.model, top_k, expand, ev_fp, effective_tp)
        if ck in cache:
            return {**cache[ck], "cached": True}

        if effective_tp:
            user_msg = PASS2_USER_TEMPLATE.format(
                query=query, context=self._format_extracted(relevant))
        else:
            user_msg = USER_TEMPLATE.format(
                query=query, n=len(evidence), context=build_context(evidence))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_msg},
        ]
        content = self._call_model(messages, temperature)
        try:
            data = _extract_json(content)
            answer = str(data.get("answer", content))
        except (json.JSONDecodeError, AttributeError):
            answer = content  # 模型没按 JSON 输出时兜底取原文
        entry = {"answer": answer, "raw": content[:500]}
        cache[ck] = entry
        save_cache(cache)
        return {**entry, "cached": False}
