"""Query 改写：用 LLM 把口语化/措辞差异大的 query 扩展成检索友好形式。

动机（实测数据驱动）
----
切换真实向量后，`param_compensation` 类 Recall@5 = 0.8250 仍是四类中偏低的，
这类问法（如「retraction is fine but still oozing」）与知识库术语措辞差异大。
改写思路：让 LLM 输出「规范缺陷术语 + 同义表述」，提高术语命中率。

策略（v2，双路分离）
----
v1 把扩展词与原 query 拼接后同时进双路，实测 Recall@5 0.8500→0.8000（负结果）：
扩展词稀释了向量 embedding 语义，也干扰了 BM25 原词命中。
v2 改为「双路分离」：
  - 向量路：仍用原 query（保护 embedding 语义）
  - BM25 路：用「原 query + 扩展术语」（词面扩展对关键词匹配有益）
本模块只负责产出扩展术语，路由由调用方决定。

设计要点
--------
1. 严格约束输出：只输出逗号分隔的英文术语短语，禁止解释，限制长度。
2. temperature=0 + 磁盘缓存（key 含 REWRITE_VERSION），保证可复现、不重复烧 token。
3. 失败降级：LLM 调用失败时返回空串（改写是增强项，不能成为故障点）。
"""
import hashlib
import json
import re
import time
from pathlib import Path

from config import DASHSCOPE_API_KEY, PROJECT_ROOT

REWRITE_MODEL = "qwen-turbo"
REWRITE_VERSION = "v2"   # v2 改双路分离策略，必须递增避免命中旧缓存
CACHE_PATH = PROJECT_ROOT / "data" / "rewrite_cache.json"

SYSTEM_PROMPT = """你是 3D 打印故障诊断检索系统的 query 改写器。
知识库是 FDM 打印质量缺陷排障手册，覆盖：挤出（不挤出/欠挤出/过挤出/堵头/研磨）、
附着（首层不粘/翘边）、表面（拉丝/渗料/层错位/振纹/鬼影/接缝/粗糙）、
结构（层分离/强度不足/尺寸偏差/孔洞）、几何（悬垂/桥接/支撑/填充）、
以及温度/回抽/流量/速度/风扇等参数校准。

任务：把用户问题改写成 2-4 个**规范英文术语短语**，用于检索。
规则：
- 只输出短语，逗号分隔，不要句子、不要解释、不要编号
- 用知识库风格的术语（如 stringing, oozing, layer shifting, warping,
  under-extrusion, over-extrusion, nozzle clog, layer adhesion, ringing,
  z-seam, elephant foot, bed adhesion, retraction, flow rate）
- 若问题已含明确缺陷词，保留它并补充同义/相关术语
- 若问题与 3D 打印质量无关（选购/软件教程/非 FDM），输出：NONE
- 总长度不超过 80 字符"""


def _cache_key(query):
    return hashlib.md5(f"{REWRITE_VERSION}::{query}".encode("utf-8")).hexdigest()


def _load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def rewrite_query(query, use_llm=True):
    """返回扩展术语串（逗号分隔短语）。失败/库外时返回空串。

    注意：本函数只产出扩展术语，不与原 query 拼接。
    调用方负责路由——向量路用原 query，BM25 路用「原 query + 扩展」。
    """
    if not use_llm or not DASHSCOPE_API_KEY:
        return ""

    cache = _load_cache()
    ck = _cache_key(query)
    if ck in cache:
        expansion = cache[ck]
    else:
        expansion = _call_llm(query)
        cache[ck] = expansion
        _save_cache(cache)

    if not expansion or expansion == "NONE":
        return ""
    return expansion


def _call_llm(query, retries=2):
    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    last_err = None
    for attempt in range(retries):
        try:
            resp = Generation.call(
                model=REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                result_format="message",
                temperature=0.0,
            )
            if resp.status_code == 200:
                text = resp.output.choices[0].message.content.strip()
                # 清洗：去掉编号/换行，压成单行逗号分隔，截断到 80 字符
                text = re.sub(r"\s+", " ", text).strip()
                return text[:80]
            last_err = f"{resp.code}: {resp.message}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    print(f"[rewrite] 调用失败，降级为原 query: {last_err}")
    return ""


def main():
    demos = [
        "retraction is fine but still oozing",
        "my corners are lifting off the bed",
        "part comes out 0.3mm undersized",
        "which filament brand should I buy",
    ]
    for q in demos:
        exp = rewrite_query(q)
        print(f"{q}\n  扩展术语: {exp or '(空)'}\n")


if __name__ == "__main__":
    main()
