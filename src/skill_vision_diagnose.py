"""P1 → P0 接线：把「缺陷图分类」翻译成 P0 生成器能直接吃的 evidence。

设计目标（硬约束：P0 检索/生成层**零改动**）
------------------------------------------
`AnswerGenerator.generate(query, evidence, ...)` 只认evidence 里
`kb_id` / `problem` / `text` 三个字段——这正是 `data/kb_chunks.json`
（56 块 kb 级 chunk）的结构。所以接线思路是：

    图像 → P1 分类 → 5 类之一 → kb_mapping 查出 KB ID
         → 从 kb_chunks.json 回查全文 chunk → evidence
         → P0 生成器原样消费（零改动）

`DiagnoseSkill` 的输出契约（agent_graph / 评测脚本都依赖它）：

    {
      "skill": "diagnose_vision",
      "rejected": bool,      # 两层拒答（视觉置信度 + P0 范围判官）任一命中
      "layer": str,          # vision_confidence / prefilter / llm / cache / none
      "reason": str,
      "evidence": [...],     # 与 P0 同构：{chunk_id,kb_id,problem,type,text}
      "kb_ids": [str],
      "top_cosine": float,   # 视觉路径无余弦相似度 → -1.0（见下）
      # ---- 视觉扩展字段（P0 侧忽略，不破坏契约）----
      "vision": {
         "n_images": int, "pred_class": str, "proba": float,
         "top2": [...], "per_image": [...],
         "vision_kb_ids": [...], "truncated": [...]
      }
    }

🔴 三条必须写下来的纪律
----------------------
1. **无图时必须逐字退回 P0**。用户只给文字描述时，本 Skill 委托
   `DiagnoseSkill.run()` 并原样返回其结果（不重排、不改字段）。
   否则 P0 的 160 题基线会被污染——而 P0 的分数是不可动的地基。

2. **`top_cosine = -1.0`，不是 0.0 也不是 1.0**。
   该字段在 P0 里的语义是「检索 top-1 余弦相似度」，全库余弦区间约
   [0.2, 0.95]。视觉路径根本没有这个量，若填0.0/1.0 会被下游
   误读成「完全不相似」或「完全相似」。-1.0 是 P0 既有的「无结果」
   取值（`results` 为空时 `top_cos = -1.0`，见 skill_diagnose.py:82），
   复用它表示「此路径不适用」最诚实。

3. **拒答取两层中最严的那个，不是最松的**。
   P1 有置信度拒答（阈值 0.90，折外标定），P0 有范围判官。
   二者独立，`rejected = vision_reject or p0_reject`。
   🔴 视觉拒答时**仍然产出 evidence**（top-2 类别的 KB），
   但 `rejected=True` 会让 agent_graph 走 reject 分支不给最终答案——
   候选证据留在返回值的 `evidence` 里供人工判断与前端展示。
   这与 `pipeline.py` 的「拒答仍输出 top-2」纪律一致。

上下文预算
----------
🔴 **这是接线里唯一的真问题**，实测（`exp/p1_p0_wiring_check.py`）：

P0 的 `MAX_CONTEXT_CHARS = 10000` 硬编码在 `answer_generator.build_context`，
不能改（P0 零回归是硬约束）。而视觉路径的 kb 级全文体量：

    Stringing         3 kb = 11,153 字Warping           3 kb = 12,191 字
    Layer_shifting    5 kb = 24,126 字     Cracking          2 kb =  7,192 字
    Off_platform      6 kb = 28,271 字

若直接丢给 `build_context`，实测会被预算挤掉整块（**先到先得的贪心**）：

    Layer_shifting 送 5 块 → 只剩 KB008（KB050/052/065/009 全丢）
    Off_platform   送 6 块 → 只剩 KB101/102/103（KB002/028/067 全丢）

**被挤掉的是次要映射与易混淆项**，且模型**不知道**它们被丢了
（build_context 只在末尾加一句「预算用尽」）→ 静默丢证据。

因此本模块在送出前自己把证据裁到预算内，规则三条：

1. **水填充（water-filling）公平分配，不贪心**。
   🔴 贪心的实测后果：Off_platform 里 KB002（13,046 字）一家吃掉 10,000 预算，
   把另外5 个 KB 全部挤掉——而 KB002 只是 *secondary*。
   改为：先给每个 KB 保底（problem 行 + 首条根因，实测 316~2,185 字），
   剩余预算按各KB **剩余体量占比**分配。这样 6 个 KB 一个不掉，
   代价是每个 KB 的根因覆盖变浅——但**「每类都有代表」比「一类讲透、
   其余全无」更符合诊断场景**（用户要的是可执行的排查清单）。
2. **降级到根因粒度**（用 kb_v2 的 causes 逐条拼），保证截断对齐语义单元。
3. **每处裁剪都追加 `[…已截断 N 条根因]`**，让模型知道有省略。
   绝不静默截断——静默会让模型以为已穷尽，与 v2.8 枚举式的设计前提冲突。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import KB_CHUNKS_PATH                      # noqa: E402
from rejector import JUDGE_MODEL                        # noqa: E402
from skill_diagnose import DiagnoseSkill               # noqa: E402
from src.vision.pipeline import DEFAULT_REJECT, diagnose as vision_diagnose  # noqa: E402

#: 视觉路径的证据预算。刻意**等于** P0 的 MAX_CONTEXT_CHARS(10000) 而非更小：
#: 送进去的已裁剪证据必须能原样通过 `build_context` 而**不触发它的截断分支**，
#: 否则 build_context 会再砍一刀，我们写在 text 里的截断标注就失去意义。
#: 实测校验见 exp/p1_p0_wiring_check.py（assert 不含「预算用尽」）。
EVIDENCE_BUDGET_CHARS = 10000

#: 每块的固定开销：header `[i] (KBxxx, retrieved) 问题: {problem}` +块间 "\n\n"。
#: 🔴 必须**实测校准**，不能拍脑袋——初版用 120 记账，实际和是 10,146 > 10,000，
#: 结果 build_context 照样触发截断（记账漂移 = 预算形同虚设）。
#: 下方 `_block_cost()` 按真实格式精确计算，这里只留这个下限做兜底。
HEADER_RESERVE = 96

#: 视觉路径无「余弦相似度」，用 -1.0 表示「此量在此路径不适用」。
#: 取值与 P0 `DiagnoseSkill` 在检索结果为空时的取值一致（skill_diagnose.py:82）。
TOP_COSINE_NA = -1.0


class VisionSkillError(RuntimeError):
    pass


class VisionDiagnoseSkill:
    """图像 +（可选）文本 → P0 evidence。契约与 DiagnoseSkill 完全对齐。"""

    name = "diagnose_vision"
    description = ("FDM 3D 打印质量缺陷诊断（图像版）：输入缺陷照片，"
                   "分类为 5 类缺陷之一并映射到知识库条目，"
                   "返回与纯文本路径同构的证据或拒答结论。")

    def __init__(self, retriever=None, text_skill: DiagnoseSkill = None,
                 budget_chars: int = EVIDENCE_BUDGET_CHARS):
        # text_skill 只在「无图」时用；显式注入便于测试时替换。
        self._text_skill = text_skill
        self.budget_chars = budget_chars
        self._kb_chunks = None   # 懒加载 chunk_id -> chunk

    # ------------------------------------------------------------ 资源
    @property
    def text_skill(self) -> DiagnoseSkill:
        if self._text_skill is None:
            self._text_skill = DiagnoseSkill()
        return self._text_skill

    def _chunks(self) -> dict:
        if self._kb_chunks is None:
            with open(KB_CHUNKS_PATH, "r", encoding="utf-8") as f:
                self._kb_chunks = {c["chunk_id"]: c for c in json.load(f)}
        return self._kb_chunks

    # ---------------------------------------------------- 证据预算裁剪
    @staticmethod
    def _cause_block(cause: dict) -> str:
        return (f"根因: {cause.get('cause', '')}\n"
                f"      方案: {cause.get('solution', '')}")

    def _block_cost(self, idx: int, kb_id: str, problem: str, text: str) -> int:
        """精确复算 build_context 里这一块的真实开销。

        必须与 answer_generator.build_context 的格式**逐字一致**：
            f"[{i}] ({kb_id}, {tag}) 问题: {problem}\n{text}"
        块间再 join("\n\n")，多一个块的代价 = 该块 + 2。
        🔴 记账宁可略高估也不能低估——低估会让 build_context 二次截断，
        把我们写在 text 里的「已截断」标注顶掉，模型就以为穷尽了。
        """
        header = len(f"[{idx}] ({kb_id}, retrieved) 问题: {problem}\n")
        return max(header + len(text) + 2, HEADER_RESERVE)

    def _render(self, kb_id: str, cause_idx: list, chunk: dict):
        """把 kb 的第 cause_idx 条根因拼成 text，附截断标注。

        返回 (text, n_truncated)。绝不静默截断。
        """
        kb = self._kb_v2().get(kb_id)
        causes = (kb or {}).get("causes") or []
        n_all = len(causes)
        shown = [causes[i] for i in cause_idx if i < n_all]
        n_cut = n_all - len(shown)

        if n_cut <= 0 and n_all:
            # 全量根因：直接用 kb_chunks 的原文（它是kb_v2 的忠实序列化，
            # 保留intro / merged_from 等信息，比自己拼更完整）
            return chunk["text"], 0

        head = kb.get("problem", chunk["problem"]) if kb else chunk["problem"]
        if not shown:                    # 保底都没拿到 → 退回原文片段
            keep = max(len(chunk["text"]) // 3, 300)
            return chunk["text"][:keep] + f"\n[…已截断，原文 {len(chunk['text'])} 字]", 0
        body = "\n".join(self._cause_block(c) for c in shown)
        tail = (f"[…已截断 {n_cut} 条根因（本条共 {n_all} 条，"
                f"已列 {len(shown)} 条）]")
        return f"{head}\n{body}\n{tail}", n_cut

    _kb_v2_cache = None

    def _kb_v2(self) -> dict:
        if self._kb_v2_cache is None:
            from config import KB_V2_PATH
            with open(KB_V2_PATH, "r", encoding="utf-8") as f:
                self._kb_v2_cache = {e["kb_id"]: e for e in json.load(f)}
        return self._kb_v2_cache

    def _build_evidence(self, ordered_ids: list):
        """max-min 公平分配预算，返回 (evidence, kb_ids_used, truncated)。

        算法（保底 + 轮转填充）：
          轮 0 保底——每个 KB 至少「problem + 首条根因 + 截断标注」；
          轮 1 起**轮转填充**——每轮给每个 KB 一次「加一条根因」的机会，
                加得起就加，直到一轮内无人能加或预算用完。

        🔴 **为什么不按比例分配预算**（试过，更差）：
            按「剩余体量占比」预切额度时，Layer_shifting 实测只花掉 6,768/10,000
            （预算剩 32% 浪费），而最重要的 primary KB008 只拿到 1 条根因
            ——因为下一条根因块比预切额度大，直接 break。
            「预算没花完 + 关键条目没覆盖」是双重损失。
            轮转填充把预算用到不能再用（6,768 → 10,000 附近），
            且 KB008 拿到多条根因。

        ⚠️ 不用纯贪心：纯贪心会让大块吃光预算把小块整个挤掉
        （实测 Off_platform 的 KB002 一家吃掉全部 10,000，其余 5 个 KB 全丢）。
        """
        chunks = self._chunks()
        # ---- 过滤索引里没有的 KB，但**记录**（不能静默丢）----
        usable, missing = [], []
        for kid in ordered_ids:
            (usable if kid in chunks else missing).append(kid)

        n = len(usable)
        if n == 0:
            return [], [], ([{"kb_id": ",".join(missing), "reason": "chunk_missing"}]
                            if missing else [])

        budget = self.budget_chars
        truncated = []
        if missing:
            truncated.append({"kb_id": ",".join(missing), "reason": "chunk_missing"})

        # ---------------- 轮 0：公平保底（每块至少 1 条根因）----------
        cause_idx = [[0] for _ in range(n)]
        cost = [0] * n

        def cost_of(i, picked):
            kid = usable[i]
            text, _ = self._render(kid, picked, chunks[kid])
            return self._block_cost(i + 1, kid, chunks[kid]["problem"], text)

        for i in range(n):
            cost[i] = cost_of(i, cause_idx[i])
        if sum(cost) > budget:
            # 保底都放不下（实测最坏 Off_platform 约 4.4k < 10k，不会发生）。
            # 仍**全部保留**——宁可超预算也不丢 KB，只记账。
            truncated.append({"reason": "floor_exceeds_budget",
                              "floor": sum(cost), "budget": budget})
            return self._finalize(usable, cause_idx, chunks, truncated)

        # ---------------- 轮 1+：轮转填充，每 KB 每轮加一条 ----------------
        max_causes = [len((self._kb_v2().get(kid) or {}).get("causes") or [])
                      for kid in usable]
        progress = True
        while progress:
            progress = False
            for i, kid in enumerate(usable):
                nxt = cause_idx[i][-1] + 1
                if nxt >= max_causes[i]:
                    continue
                trial = cause_idx[i] + [nxt]
                c = cost_of(i, trial)
                if sum(cost) - cost[i] + c <= budget:
                    cause_idx[i], cost[i] = trial, c
                    progress = True

        return self._finalize(usable, cause_idx, chunks, truncated)


    def _finalize(self, usable, cause_idx, chunks, truncated):
        """渲染最终 evidence 并汇总截断记录。"""
        evidence, used_ids = [], []
        for i, kid in enumerate(usable):
            text, n_cut = self._render(kid, cause_idx[i], chunks[kid])
            evidence.append({
                "chunk_id": kid, "kb_id": kid,
                "problem": chunks[kid]["problem"], "type": chunks[kid]["type"],
                "text": text, "expanded": False, "source": "vision",
                "n_causes_shown": len(cause_idx[i]),
            })
            used_ids.append(kid)
            if n_cut and not any(t.get("kb_id") == kid for t in truncated):
                truncated.append({"kb_id": kid, "truncated_causes": n_cut})
        return evidence, used_ids, truncated



    # ------------------------------------------------------------ 拒答
    @staticmethod
    def _p0_reject(query, top_k, use_judge, model):
        """跑 P0 的两层拒答，返回 (rejected, layer, reason, top_cos)。

        直接复用 `DiagnoseSkill.run` 的判官链路，但**不复用它的检索**——
        视觉路径的 evidence 是给定的，不需要再检索一遍。
        ⚠️ 这里刻意只借判官函数（prefilter / judge_once / 缓存），
        不调 retriever.retrieve，避免多烧一次 embedding + BM25。
        """
        from rejector import (prefilter_reject, judge_once,
                             _cache_key, load_cache, save_cache)
        hit, r = prefilter_reject(query)
        if hit:
            return True, "prefilter", r, TOP_COSINE_NA
        cache = load_cache()
        ck = _cache_key(query, model)
        if ck in cache:
            v, layer = cache[ck], "cache"
        else:
            v = judge_once(query, [], model=model)
            cache[ck] = {"answerable": v["answerable"], "reason": v["reason"]}
            save_cache(cache)
            layer = "llm"
        if v["answerable"]:
            return False, layer, v["reason"], TOP_COSINE_NA
        return True, layer, v["reason"], TOP_COSINE_NA

    # ------------------------------------------------------------ run
    def run(self, query, image_paths=None, top_k=5, use_judge=True,
            model=JUDGE_MODEL, reject_threshold: float = DEFAULT_REJECT,
            device: str = "cuda", with_confusable: bool = True,
            model_dir=None, fallback_to_text: bool = True):
        """图像诊断主入口。

        参数
        ----
        query        : 用户文字描述。图像路径下作为「附加上下文」，
                      用于P0 范围判官（判这个提问是否在知识库覆盖范围内）。
        image_paths  : 图片路径列表。为空 → **完全委托 P0 文本路径**。
        reject_threshold : P1 置信度拒答阈值（默认 0.90，折外标定值）。

        ⚠️ `query` 为空字符串时不做 P0 判官——没有提问就没有「范围问题」，
        此时视觉拒答是唯一判据。
        """
        # ---------- 分支 1：无图 → 逐字退回 P0（纪律①）----------
        if not image_paths:
            if not fallback_to_text:
                raise VisionSkillError("无图且 fallback_to_text=False")
            out = self.text_skill.run(query, top_k=top_k, use_judge=use_judge)
            # 标记来源，便于审计（不改动P0 的任何一个字段值）
            out = {**out, "skill": self.name, "route": "text",
                   "vision": {"n_images": 0, "pred_class": None, "proba": None,
                              "top2": [], "per_image": [],
                              "vision_kb_ids": [], "truncated": []}}
            return out

        # ---------- 分支 2：有图 → P1 视觉链路 ----------
        preds = vision_diagnose(image_paths, reject_threshold=reject_threshold,
                                device=device, with_confusable=with_confusable,
                                model_dir=model_dir)
        if not preds:
            raise VisionSkillError("vision_diagnose 返回空结果（图片列表为空？）")

        # 多图：按置信度降序——高置信的那张说了算
        ranked = sorted(preds, key=lambda p: -p.proba)
        primary = ranked[0]

        # 🔴 **只用主图的类别映射，不合并所有图的 KB**。
        # 初版把每张图的 kb_ids 依次拼接，实测传 3 张不同类别的图时
        # evidence 里混进 **12 个 KB / 3 个互斥类别**的根因——
        # 生成器会同时给出「拉丝」「层分离」「不粘床」三套互斥建议，
        # 用户无从下手。**类别是互斥的诊断结论，不是可以并集的信息。**
        # 多图的价值在于「同一缺陷多角度互相印证」，不在于「多缺陷一次性诊断」。
        ordered_ids = list(primary.kb_ids)
        seen = set(ordered_ids)

        # 类别不一致要**显式告知**（不是静默取最高那张）。
        # 真实场景两种成因：① 用户一次传了不同部件/不同缺陷的照片
        #   → 应拆成多次诊断；② 分类器对某张图没把握（但 proba 高说明不太可能）。
        # 无论哪种，用户都必须知道「我看到的不止一种东西」。
        class_conflict = sorted({p.pred_class for p in ranked})
        conflict_note = ""
        if len(class_conflict) > 1:
            per = "；".join(f"{Path(p.source_path).name}→{p.pred_class}"
                            f" {p.proba:.2f}" for p in ranked)
            conflict_note = (f"⚠️ {len(preds)} 张图给出了 {len(class_conflict)} 个不同类别"
                             f"（{per}）。本次只按置信度最高的一张（"
                             f"{primary.pred_class}）给建议；若要分别诊断，"
                             f"请一次只传一张。")

        evidence, used_ids, truncated = self._build_evidence(ordered_ids)

        # ---------- 拒答：两层取严（纪律③）----------
        vision_rejected = any(p.rejected for p in preds)
        p0_rejected, p0_layer, p0_reason, _ = (False, "none", "", -1.0)
        if use_judge and (query or "").strip():
            p0_rejected, p0_layer, p0_reason, _ = self._p0_reject(
                query, top_k, use_judge, model)

        if vision_rejected:
            layer = "vision_confidence"
            bad = [p for p in preds if p.rejected]
            reason = ("视觉置信度不足：" + "；".join(
                f"{Path(p.source_path).name} 判为 {p.pred_class} "
                f"概率仅 {p.proba:.4f}" for p in bad))
            rejected = True
        elif p0_rejected:
            layer, reason, rejected = p0_layer, p0_reason, True
        else:
            layer = p0_layer if p0_layer != "none" else "none"
            reason = p0_reason or "(vision path, 未触发拒答)"
            rejected = False

        return {
            "skill": self.name,
            "route": "vision",
            "rejected": rejected,
            "layer": layer,
            "reason": reason,
            # 🔴 拒答时 evidence **仍非空**：top-2 候选留给人工判断（纪律③）
            "evidence": evidence,
            "kb_ids": used_ids,
            "top_cosine": TOP_COSINE_NA,
            "vision": {
                "n_images": len(preds),
                "pred_class": primary.pred_class,
                "proba": round(primary.proba, 4),
                "top2": [(c, round(q, 4)) for c, q in primary.top2],
                "kb_details": primary.kb_details,
                "per_image": [p.to_dict() for p in preds],
                "vision_kb_ids": ordered_ids,
                "kb_ids_used": used_ids,
                "class_conflict": class_conflict,
                "conflict_note": conflict_note,
                "truncated": truncated,
                "reject_threshold": reject_threshold,
                "vision_rejected": vision_rejected,
                "p0_rejected": p0_rejected,
                "cv_macro_f1_note": ("P1 折外 macro-F1 99.73%（DC-02R 内、"
                                     "group-aware 折、5 类）；"
                                     "**不可外推到换机器/换材料**"),
            },
        }


def format_vision_report(skill_out: dict) -> str:
    """人可读摘要（给 CLI / 演示用）。"""
    v = skill_out.get("vision", {})
    L = [f"route={skill_out['route']}  rejected={skill_out['rejected']} "
         f"layer={skill_out['layer']}"]
    if skill_out["route"] == "vision":
        L.append(f"判定类别: {v['pred_class']}  概率 {v['proba']}")
        L.append("Top-2: " + "  |  ".join(f"{c} {q}" for c, q in v["top2"]))
        L.append(f"映射 KB: {', '.join(skill_out['kb_ids'])}")
        if v.get("truncated"):
            L.append(f"⚠️ 证据裁剪: {v['truncated']}")
    L.append(f"理由: {skill_out['reason']}")
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="缺陷图路径（可多张）")
    ap.add_argument("--query", default="", help="附加文字描述（可选）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reject-threshold", type=float, default=DEFAULT_REJECT)
    ap.add_argument("--budget", type=int, default=EVIDENCE_BUDGET_CHARS)
    a = ap.parse_args()

    sk = VisionDiagnoseSkill(budget_chars=a.budget)
    out = sk.run(a.query or "这张打印出的模型照片里是什么缺陷？该怎么修？",
                 image_paths=a.images or None,
                 reject_threshold=a.reject_threshold, device=a.device)
    print(format_vision_report(out))
    for e in out["evidence"]:
        print(f"\n[{e['chunk_id']}] {e['problem']}\n{e['text'][:300]}…")


if __name__ == "__main__":
    main()
