"""拒答判官：用 LLM 判断用户问题是否落在知识库覆盖范围内。

背景
----
实测证明 **bi-encoder 相似度阈值无法做拒答**：

    DashScope 余弦 top-1  正例 0.5638 ~ 0.8628（mean 0.7088）
                          负例 0.4771 ~ 0.7502（mean 0.6376）
                          分离间隔 = -0.1865  → 分布重叠，无干净切分点

根因：库外负例（如「3D printer mainboard upgrade」）与库内内容**同属 3D 打印
领域**，语义距离天然近。而知识库只覆盖「打印质量缺陷排障」这一子域。

因此改用 LLM 做 **范围判定（scope check）**：
判断「**用户问题本身的类型**是否在知识库覆盖范围内」。

⚠️ 判据必须是「问题类型是否在覆盖范围内」，**不是**「检索片段是否回答了问题」。
后者会把「库内问题但召回不准」误判成库外，造成大量误拒——
第一版 prompt 就是这么写的，误拒率一度高达 45%（详见 skill
`rag-retrieval-diagnostics`「坑 4」）。

设计要点
--------
1. 判据写清楚「知识库只覆盖 FDM 打印质量缺陷排障」，把设备/软件/材料/工艺
   科普/商业等划为库外。
2. **两步判据（v3 新增）**——实测发现单层「话题是否在范围内」会被词面命中带偏：
   - 第一步判**意图**：问的是「已出现的质量现象」（排障）还是
     「工具/软件/选型/事前参数」（库外）。**动宾结构**是分界线——
     `fix/unclog/prevent + 缺陷` → 库内；`use/set/configure + 软件`、
     `best/which + 设备材料` → 库外。
     ⚠️ 不能用 `how to` 当特征：`how to unclog a nozzle` 与
     `how to use supports in slicer` 句式相同、归属相反。
   - 第二步判**现象**是否被覆盖，并明确「检索召回不准 ≠ 该拒答」。
3. **机械类根因归入覆盖范围**（背隙/回差、皮带张紧、微步、丝杆定位）：
   知识库原文把它们写成「某质量缺陷的原因」（KB019 侧壁纹路、KB026 尺寸偏差），
   所以问这些根因判 answerable；而「设备改装/接线/固件」仍属库外。
4. 明确保护「口语化 / 含否定条件」的合法提问（如「retraction is fine but
   still oozing」），避免误拒正例。
5. 检索片段只作参考、**明确标注不作为判断依据**。
6. 结果落盘缓存，同一 query 不重复调用；cache key 含 PROMPT_VERSION。
7. **两层结构**：Layer 1 确定性预过滤（`prefilter_reject`，纯规则、零 API）→
   Layer 2 LLM 范围判官（本文件的 prompt）。
8. 复现性（实测见 `exp/exp_judge_reproducibility.py` / `data/judge_repro_result.json`）：
   **prompt v8** 下，temp=0.0 跑 200 条 × 5 次 → 标签一致率 **200/200 = 100.00%**、
   F1 五次全为 **0.9873（零波动）**、误拒正例恒为 **0 条**。
   > 对比 v2 时代：一致率 99.00%、F1 区间 [0.9351, 0.9487]（宽 0.0136）。
   > **改两步判据后方差归零** → 说明此前的不稳定来自**判据模糊**（模型在两种解读间摇摆），
   > 而不是 temperature 或模型随机性。
9. 留出集（`exp/exp_holdout_scope.py`）：40 + 24 条手写集均 **100.00%**。
   ⚠️ 同为作者手写，不能替代第三方标注。
"""
import hashlib
import json
import re
import time
from pathlib import Path

from config import DASHSCOPE_API_KEY, PROJECT_ROOT

JUDGE_MODEL = "qwen-turbo"
PROMPT_VERSION = "v8"          # 改 prompt 时必须递增，否则会命中旧缓存
CACHE_PATH = PROJECT_ROOT / "data" / "reject_cache.json"

SYSTEM_PROMPT = """你是 3D 打印故障诊断知识库的**范围审核员**。

该知识库是一份 **FDM 打印质量缺陷排障手册**。覆盖范围：

- 挤出类：不挤出、欠挤出、过挤出、堵头、料线研磨、流量异常
- 附着类：首层不粘床、翘边、边角翘起、附着力不足
- 表面类：拉丝/渗料、层错位、振纹（ringing）、鬼影（ghosting）、
  接缝/疤痕（z-seam）、表面粗糙、抹痕
- 结构类：层分离（delamination）、强度不足、尺寸偏差、孔洞
- 几何类：悬垂质量差、桥接下垂/拉丝、支撑难拆、填充异常
- **打印参数与校准**：温度塔校准、挤出量校准、回抽、流量倍率、层高、
  速度、风扇、Z 缝对齐等——⚠️ **知识库给出的修复方案本身就是这些参数**，
  所以「某参数该设多少 / 怎么校准」属于覆盖范围。
- 上述缺陷的**机械类根因**：背隙/回差（backlash）、皮带与导轨张紧、
  微步（micro-stepping）设置、Z 轴丝杆定位——知识库把它们写成
  「某质量缺陷的原因」（如尺寸偏差、侧壁纹路），属于覆盖范围。

**不覆盖**：设备选型与硬件改装（换主板/换喷嘴/加装外罩）、接线与固件、
CAD 建模与切片软件**功能操作教程**、材料与配件选购/存储/回收、
非 FDM 工艺（SLA/DLP/树脂/激光）、工艺科普（FDM vs SLA、DLP vs SLA）、
商业运营、后处理与上色装饰。"""

USER_TEMPLATE = """用户问题：
{query}

系统按语义检索到的知识库片段（供你了解知识库内容，**不作为判断依据**）：
{context}

---

请按**两步**判断。

### 第一步 · 判「意图」——这个问题锚定到「打印质量」了吗？

**库内（answerable）**：只要问题的落点是 **FDM 打印质量**，
**无论句式是「现象+怎么办」还是「某参数/校准该怎样」**，都算在范围内。
因为知识库的修复方案本身就是参数（层高、温度、速度、回抽、流量、风扇、Z 缝）。
- 现象式：`my corners are lifting`、`my part comes out 0.3 mm undersized`
- 参数式（**同样算库内**）：`nozzle temperature for PETG`、
  `retraction distance to stop stringing`、`calibrating the extrusion multiplier`、
  `jerk settings causing layer shift`
  ⚠️ 判断依据是它**锚定到某个打印质量/缺陷**（stringing、layer shift、挤出量…），
  **不是**它「有没有先描述现象」。

**库外（unanswerable）**：落点**不是打印质量本身**，而是——
- **切片/CAD 软件的功能操作教程**：how to use / set / configure **某软件功能**
  （摆放支撑、导入模型、界面操作…）——⚠️ 注意与「参数值该设多少」的区别：
  前者是**软件怎么用**，后者是**工艺参数是多少**，后者算库内。
- **「择优/选购」形态**：`best X`、`the best X`、`which X should I get / use` ——
  **在候选项之间挑一个**，且**没有「已经出了什么毛病」的前提**。
  例：`which filament brand should I buy`、`what is the best build surface`、
  `best nozzle diameter for fine detail`
  ⚠️ 与库内的对照：`which settings affect layer shifting`
  （这是**在追问某缺陷的成因**）→ 库内；
  `best X for Y`（在优化一个不存在的问题）→ 库外。

  **🔴 硬规则（先执行这条再考虑其他）**：
  以 `best` / `which` 开头、或形如 `best X` / `the best X` 的问题，
  先看它是不是在问「**哪个更好 / 该选哪个**」——
  是 → **直接判 unanswerable**，除非句子里**本身描述了已出现的缺陷现象**
  （如 `which setting is causing my layer shift`）。
  **绝对不要**因为 X 与某缺陷「话题相关」就把它算成库内：
  `best build surface` 虽与附着有关，但问的是**买哪种** → 库外；
  `best infill pattern for stronger parts` 是**择优** → 库外。
  这类问题的答案在**选购指南**里，不在**排障手册**里。
- **「索取参考资料/标准数据」形态**——用户在要**一份文档、标准或数据**，
  而不是在排查已发生的问题 → **unanswerable**。
  知识库是**排障手册**，既不是教程，也不是标准/规格参考。
  典型：`X guide` / `X tutorial` / `X basics` / `X handbook` / `overview` /
  `cheat sheet` / `X chart` / `X table` / `spec sheet` / `glossary` /
  `tolerance chart` / `comparison table`
  ⚠️ 判据是**「他要的是资料，不是解决方案」**，不要靠单个词——
  也不要因为资料主题（tolerances / dimensions…）与某缺陷相关就判库内。
- **硬件本身**：升级/改装/换件/接线/固件
- 非 FDM 工艺（SLA/DLP/树脂）、工艺科普、商业运营、后处理上色

⚠️ **话题命中 ≠ 在范围内**：问题里出现了 supports / build plate / infill 等词，
但实际在问「**软件功能怎么用**」「**该买哪个**」，仍判 **unanswerable**。

### 第二步 · 判「现象/参数」是否被覆盖

- 属于上面列出的类别（含**打印参数与校准**、**机械类根因**）→ **answerable**
- 属于该范围、但检索片段没匹配准（库里没写、措辞差异导致召回偏差）
  → **仍是 answerable**（这是检索质量问题，**不该拒答**）
- 措辞口语化、或带否定条件（如 `sagging but not warping`、`no clog but stops extruding`）
  只要能对应到某类打印缺陷，都判 **answerable**

只输出 JSON，不要任何其他内容：
{{"answerable": true, "reason": "一句话理由"}}"""


def build_context(results, max_items=5):
    """把检索结果拼成判官可读的上下文。"""
    lines = []
    for i, r in enumerate(results[:max_items], 1):
        lines.append(f"{i}. [{r['kb_id']}] {r['problem']}")
        if r.get("text"):
            lines.append(f"   {r['text'][:120]}")
    return "\n".join(lines) if lines else "（无检索结果）"


def _extract_json(text):
    """从模型输出里稳健地抠出 JSON。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def judge_once(query, results, model=JUDGE_MODEL, api_key=None, retries=3,
               temperature=0.0):
    """对单条 query 做一次判官调用（**仅 Layer 2，不含预过滤**）。返回 {answerable, reason, raw}。

    temperature: 默认 0.0（生产配置，追求可复现）。
                 复现性实验会显式传非零值做压力测试。

    ⚠️ 本函数**有意不包含预过滤**——复现性实验要测的正是「LLM 这一层的稳定性」，
       若在这里短路，实验就失去意义。生产链路请走 `judge_many` / `answer_or_reject`。
    """
    import dashscope
    from dashscope import Generation

    key = api_key or DASHSCOPE_API_KEY
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set，无法调用判官")
    dashscope.api_key = key

    user_msg = USER_TEMPLATE.format(query=query, context=build_context(results))
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                result_format="message",
                temperature=temperature,
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                data = _extract_json(content)
                return {
                    "answerable": bool(data.get("answerable", True)),
                    "reason": str(data.get("reason", ""))[:200],
                    "raw": content[:300],
                }
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"判官调用失败: {last_err}")


# ---------------- Layer 1：确定性预过滤（无 LLM / 零成本 / 完全可复现） ----------------
# 知识库是「排障手册」，不是教程、指南或标准参考资料。
# 当用户在**索取一份资料/文档/标准数据**时，无需调用 LLM 即可判定库外。
#
# 为什么需要这一层：LLM 判官在「公差/规格」这类问题上会被话题牵引
# （`公差` → `尺寸偏差` → 误判库内），实测在**两份不同集子上各出现 1 次**，
# 属**稳定缺陷**而非随机波动；改 prompt 措辞 3 轮均未修好（v7/v8），
# 故下沉为确定性规则。
#
# ⚠️ 该规则已在 3 份集子上做过误伤扫描：
#    eval_set 200 + 留出集#1 40 + 留出集#2 24，命中 7 条**全部为负例**，正例误伤 0 条。
_DOC_NOUN_RE = re.compile(
    r"\b(guide|guides|tutorial|tutorials|basics|handbook|manual|faq|glossary|"
    r"cheat ?sheet|overview|datasheet|data ?sheet|spec ?sheet|specification|"
    r"chart|charts|comparison ?table|checklist|wiki|blog|article|documentation|docs?)\b",
    re.I,
)
PREFILTER_ENABLED = True


def prefilter_reject(query):
    """Layer 1 预过滤。返回 (是否拒答, 理由)。纯规则、无 API 调用。"""
    if not PREFILTER_ENABLED:
        return False, ""
    m = _DOC_NOUN_RE.search(query or "")
    if m:
        return True, (f"[预过滤] 问题在索取资料/指南/标准数据（命中 “{m.group(0)}”），"
                      f"知识库是排障手册、不含此类内容")
    return False, ""


def _cache_key(query, model):
    # 含 PROMPT_VERSION：改 prompt 后自动失效旧缓存
    return hashlib.md5(f"{PROMPT_VERSION}::{model}::{query}".encode("utf-8")).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_many(pairs, model=JUDGE_MODEL, verbose=True, save_every=10):
    """批量判官：**Layer 1 预过滤 → Layer 2 LLM 判官**（带磁盘缓存）。

    pairs: [(query, results), ...]
    返回: [ {answerable, reason, cached, layer}, ... ]
          layer: "prefilter" = 规则命中；"llm" = LLM 判官；"cache" = 缓存命中
    """
    cache = load_cache()
    out = []
    for i, (query, results) in enumerate(pairs, 1):
        # ---- Layer 1：确定性预过滤（不调 API、不进缓存） ----
        hit, reason = prefilter_reject(query)
        if hit:
            out.append({"answerable": False, "reason": reason,
                        "cached": False, "layer": "prefilter"})
            if verbose:
                print(f"  [{i}/{len(pairs)}] prefiltered", end="\r")
            continue

        # ---- Layer 2：LLM 判官（带缓存） ----
        ck = _cache_key(query, model)
        if ck in cache:
            out.append({**cache[ck], "cached": True, "layer": "cache"})
            if verbose:
                print(f"  [{i}/{len(pairs)}] cache hit", end="\r")
            continue
        res = judge_once(query, results, model=model)
        cache[ck] = {"answerable": res["answerable"], "reason": res["reason"]}
        out.append({**res, "cached": False, "layer": "llm"})
        if verbose:
            print(f"  [{i}/{len(pairs)}] judged", end="\r")
        if i % save_every == 0:
            save_cache(cache)
    save_cache(cache)
    if verbose:
        print()
    return out


def threshold_baseline(results, cosine_threshold=0.6865):
    """对比基线：纯余弦阈值法（实测最优阈值 0.6865，F1 仅 0.53）。"""
    top_cos = results[0]["cosine"] if results else -1.0
    return top_cos >= cosine_threshold


def answer_or_reject(retriever, query, top_k=5, use_judge=True, model=JUDGE_MODEL):
    """完整 RAG 链路：检索 → **预过滤 → 判官** → 返回结果或拒答。

    P1 / P2 的 Agent 可直接调用本函数作为「知识检索」这一环。

    返回:
        {
          "rejected": bool,          # True = 拒答
          "results": [...],          # 拒答时为空列表
          "reason": str,             # 拒答/放行理由
          "layer": str,              # prefilter / llm / none
          "top_cosine": float,       # 最高余弦相似度（可观测性）
        }
    """
    results = retriever.retrieve(query, top_k=top_k)
    top_cos = results[0]["cosine"] if results else -1.0

    if not use_judge:
        return {"rejected": False, "results": results, "layer": "none",
                "reason": "(judge disabled)", "top_cosine": top_cos}

    # Layer 1
    hit, reason = prefilter_reject(query)
    if hit:
        return {"rejected": True, "results": [], "layer": "prefilter",
                "reason": reason, "top_cosine": top_cos}

    # Layer 2
    v = judge_once(query, results, model=model)
    if v["answerable"]:
        return {"rejected": False, "results": results, "layer": "llm",
                "reason": v["reason"], "top_cosine": top_cos}
    return {"rejected": True, "results": [], "layer": "llm",
            "reason": v["reason"], "top_cosine": top_cos}
