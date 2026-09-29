"""P2 诊断 Skill：把 P0/P1 的「检索 + 拒答」链路封装成标准 Skill 接口。

Skill 契约（Agent 下所有 Skill 统一遵守）：
    name / description : 供编排层路由与日志展示
    run(query, **kw)   : 返回可序列化的结构化 dict

本 Skill 的输出：
    {
      "skill": "diagnose",
      "rejected": bool,      # 拒答链路（预过滤 → 判官）是否命中
      "layer": str,          # prefilter / llm / cache / none
      "reason": str,         # 拒答或放行理由
      "evidence": [...],     # 全文证据（无 200 字截断），供下游生成
      "kb_ids": [str],       # 涉及知识库条目（去重保序）
      "top_cosine": float,   # 可观测性指标
    }

设计要点
--------
1. 证据用**全文**：HybridRetriever.retrieve 的 text 截断到 200 字是给展示用的，
   生成答案必须喂全文，这里按 chunk_id 回查完整 chunk。
2. **Parent-Child 扩展**：检索粒度是 cause 级（细，利于精准匹配），
   返回时对 top-1 kb_id 补齐同 problem 下的其余 cause 兄弟块
   （标记 expanded=True）。生成器拿到的是「最可能问题的完整诊断视图」，
   而不是孤立的单个根因——等价于轻量 parent-child 分块，零额外 API。
3. 拒答判官复用 rejector 的磁盘缓存（同 key 规则），批量评测不重复烧 token。
"""
import json

from config import KB_V2_PATH, EXPAND_PARENT
from hybrid_retriever import HybridRetriever
from rejector import (
    prefilter_reject, judge_once, _cache_key, load_cache, save_cache,
    JUDGE_MODEL,
)


class DiagnoseSkill:
    name = "diagnose"
    description = ("FDM 3D 打印质量缺陷诊断：输入打印问题描述，"
                   "执行混合检索与范围拒答，返回知识库证据或拒答结论。")

    def __init__(self, retriever=None, expand_parent=None):
        self.retriever = retriever or HybridRetriever()
        # expand_parent 为 None 时读 config.EXPAND_PARENT（env 可覆盖，检索精度实验用）
        self.expand_parent = expand_parent if expand_parent is not None else EXPAND_PARENT

    def _kb(self):
        if self._kb_by_id is None:
            with open(KB_V2_PATH, "r", encoding="utf-8") as f:
                self._kb_by_id = {e["kb_id"]: e for e in json.load(f)}
        return self._kb_by_id

    def _full_chunk(self, chunk_id):
        c = self.retriever.chunks[self.retriever.chunk_id_to_idx[chunk_id]]
        return {k: c.get(k) for k in
                ("chunk_id", "kb_id", "problem", "type", "cause", "text")}

    def _expand_parent(self, kb_id, have_ids):
        """Parent-Child 扩展：补齐某 problem 下未命中的兄弟 cause 块。"""
        out = []
        if kb_id not in self._kb():
            return out
        for i in range(len(self._kb()[kb_id]["causes"])):
            cid = f"{kb_id}_c{i}"
            if cid in have_ids or cid not in self.retriever.chunk_id_to_idx:
                continue
            item = self._full_chunk(cid)
            item["expanded"] = True
            out.append(item)
        return out

    def run(self, query, top_k=5, use_judge=True, model=JUDGE_MODEL):
        results = self.retriever.retrieve(query, top_k=top_k)
        top_cos = results[0]["cosine"] if results else -1.0

        # ---- 拒答链路：Layer 1 预过滤 → Layer 2 判官（带磁盘缓存）----
        rejected, layer, reason = False, "none", "(judge disabled)"
        if use_judge:
            hit, r = prefilter_reject(query)
            if hit:
                rejected, layer, reason = True, "prefilter", r
            else:
                cache = load_cache()
                ck = _cache_key(query, model)
                if ck in cache:
                    v, layer = cache[ck], "cache"
                else:
                    v = judge_once(query, results, model=model)
                    cache[ck] = {"answerable": v["answerable"],
                                 "reason": v["reason"]}
                    save_cache(cache)
                    layer = "llm"
                if v["answerable"]:
                    reason = v["reason"]
                else:
                    rejected, reason = True, v["reason"]

        if rejected:
            return {"skill": self.name, "rejected": True, "layer": layer,
                    "reason": reason, "evidence": [], "kb_ids": [],
                    "top_cosine": top_cos}

        # ---- 证据组装：全文 chunk + parent-child 扩展 ----
        evidence, have = [], set()
        for r in results:
            item = self._full_chunk(r["chunk_id"])
            item.update({"cosine": r["cosine"], "bm25_score": r["bm25_score"],
                         "score": r["score"], "expanded": False})
            evidence.append(item)
            have.add(r["chunk_id"])
        if self.expand_parent and evidence:
            # 只扩 top-1 命中 problem 的兄弟块。
            # ⚠️ 曾试扩 top-3 个去重 kb（max_expand_kb=3），端到端任务成功率
            # 从 15.6% 降至 14.4%（multi_defect 5→4）——检索命中相关 kb 时才
            # 扩展才有意义，而 multi_defect 的第二个 gold kb(如 KB007)常未被召回，
            # 扩命中的“无关 problem”只会稀释信息。故回退为只扩 top-1。
            evidence.extend(self._expand_parent(evidence[0]["kb_id"], have))

        kb_ids = list(dict.fromkeys(e["kb_id"] for e in evidence))
        return {"skill": self.name, "rejected": False, "layer": layer,
                "reason": reason, "evidence": evidence, "kb_ids": kb_ids,
                "top_cosine": top_cos}


def main():
    skill = DiagnoseSkill()
    for q in ["my print has stringing", "best filament brand to buy"]:
        out = skill.run(q)
        print(f"\n=== {q} ===")
        print(f"rejected={out['rejected']} layer={out['layer']} "
              f"kb_ids={out['kb_ids']}")
        for e in out["evidence"][:3]:
            tag = "expanded" if e.get("expanded") else "retrieved"
            print(f"  [{e['chunk_id']}] ({tag}) {e['text'][:60]}...")


if __name__ == "__main__":
    main()
