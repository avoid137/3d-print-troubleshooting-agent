"""P0↔P1 编排层：按「有没有图」自动选路，图文双模态诊断。

设计约束
--------
🔴 **`src/agent_graph.py` 与 `src/skill_diagnose.py` 属P0 冻结区，一行不改**
（纪律：P1 不改 P0 检索/生成层，重做必回归）。本模块是**新增的适配层**：

    图/ 图+文 ──→ VisionDiagnoseSkill ──→ evidence ──┐
                                                      ├─→ AnswerGenerator
    纯文本 ─────→ DiagnoseSkill ────────→ evidence ──┘

复用 P0 已有的 `_make_nodes`（同一套 generate / reject 逻辑），
只替换「diagnose 节点用哪个 Skill」，因此两路行为天然一致。

图拓扑（与 `agent_graph` 完全相同的形状）：

    START → diagnose →(条件边)→ generate → END
                        └(拒答)→  reject   → END

拒答
----
两层拒答（视觉置信度 + P0 范围判官）都在 Skill 内完成，
本模块只读 `rejected` 标志做条件边分流——与 P0 的处理方式一致。

拒答时的回复
------------
纯文本拒答沿用 P0 的 `REJECT_REPLY`（一字不改）。
视觉拒答用 `VISION_REJECT_REPLY`：**带上 top-2 类别与建议 KB**，
因为视觉拒答往往不是「问题在库外」，而是「这张图我看不准」——
只回一句「无法回答」对用户毫无帮助，等于把可执行信息全丢掉。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import StateGraph, START, END          # noqa: E402

from agent import REJECT_REPLY                               # noqa: E402,F401
from agent_graph import AgentState, _make_nodes as _p0_make_nodes  # noqa: E402
from answer_generator import AnswerGenerator, ANSWER_MODEL   # noqa: E402
from skill_diagnose import DiagnoseSkill                    # noqa: E402
from skill_vision_diagnose import VisionDiagnoseSkill        # noqa: E402
from src.vision.pipeline import DEFAULT_REJECT               # noqa: E402

#: 视觉路径需要透传给 skill.run 的额外参数（来自 VisionAgentState）
VISION_STATE_KEYS = ("image_paths", "reject_threshold", "device",
                     "with_confusable", "model_dir")


class VisionAgentState(AgentState, total=False):
    """在 P0 的 AgentState 上**追加**视觉字段（不改 P0 原文件）。

    🔴 必须显式声明：`AgentState` 是 `TypedDict(total=False)`，
    而 **LangGraph 会静默丢弃 schema 里没声明的入参**——
    实测（最小复现）：
        StateGraph(S2).invoke({'query':..,'image_paths':[..]}) → {'query': ..}
    `image_paths` 直接消失 → 有图也走文本路径，且**不报任何错**。
    这种「静默失效」正是本项目防线要防的那类 bug，
    所以这里补全状态字段，并在验收里断言 `route == "vision"`。
    """
    image_paths: list
    reject_threshold: float
    device: str
    with_confusable: bool
    model_dir: object


def route_after_diagnose(state: VisionAgentState) -> str:
    return "reject" if state["rejected"] else "generate"


def make_nodes(skill, generator):
    """在 P0 节点工厂外层包一层，把 state 里的视觉参数传给 skill.run。

    🔴 P0 的 `_make_nodes` 生成的 diagnose 只传 query/top_k/use_judge。
    这里**不修改**它，而是另写一个 diagnose 节点：
      - 无图 → 直接调 P0 的原节点（保证文本路径逐字一致，不做任何包装）
      - 有图 → 调视觉 skill，返回字段与 P0 同构（trace 结构也保持一致）
    generate / reject 两个节点**直接复用 P0 原件**，不重新实现。
    """
    base_diagnose, generate, reject = _p0_make_nodes(skill, generator)

    def diagnose(state: VisionAgentState) -> dict:
        if not state.get("image_paths"):
            return base_diagnose(state)          # 纯文本 → P0 原节点

        extra = {k: state[k] for k in VISION_STATE_KEYS if k in state}
        sk = skill.run(state["query"],
                       top_k=state.get("top_k", 5),
                       use_judge=state.get("use_judge", True),
                       **extra)
        return {
            "skill_out": sk, "rejected": sk["rejected"], "layer": sk["layer"],
            "kb_ids": sk["kb_ids"], "top_cosine": sk["top_cosine"],
            "trace": [{
                "node": "diagnose", "route": sk.get("route"),
                "rejected": sk["rejected"], "layer": sk["layer"],
                "reason": sk["reason"], "kb_ids": sk["kb_ids"],
                "n_evidence": len(sk["evidence"]),
                "top_cosine": sk["top_cosine"],
                "vision": {k: sk["vision"].get(k)
                           for k in ("n_images", "pred_class", "proba")},
            }],
        }

    return diagnose, generate, reject


#: 视觉拒答的回复模板。带上 top-2 类别与建议 KB，
#: 因为「看不准」≠「库里没有」，直接复用 P0 的 REJECT_REPLY 会丢掉可执行信息。
VISION_REJECT_REPLY = (
    "抱歉，这张缺陷照片我没能可靠识别（置信度不足），"
    "不确定属于哪一类，因此先不给确定性的调整建议——"
    "照着错的建议去改参数反而会帮倒忙。\n"
    "候选方向：{candidates}\n"
    "可以试：① 用手机贴近拍缺陷部位（只拍缺陷件，别拍整台机器）；"
    "② 光线侧打、避免反光；③ 或者直接描述你看到的现象（翘边/拉丝/层错位…），我按文字诊断。\n"
    "对应知识库条目（供参考，不保证适用）：{kb_ids}"
)


def _format_vision_reject(skill_out: dict) -> str:
    v = skill_out.get("vision", {})
    cands = " / ".join(f"{c} {q:.2f}" for c, q in v.get("top2", [])) or "（无）"
    return VISION_REJECT_REPLY.format(candidates=cands,
                                      kb_ids=", ".join(skill_out.get("kb_ids", [])) or "（无）")


class VisionDiagnoseAgent:
    """对外接口与 `agent_graph.GraphDiagnoseAgent.answer` 对齐，多一个 image_paths。

    用法::

        ag = VisionDiagnoseAgent()
        ag.answer("my print has stringing")                    # 纯文本 → P0 原路径
        ag.answer("这图是什么问题？", image_paths=["a.jpg"])     # 图像→ P1 路径
    """

    def __init__(self, text_skill=None, vision_skill=None,
                 generator=None, model=ANSWER_MODEL):
        self.text_skill = text_skill or DiagnoseSkill()
        self.vision_skill = vision_skill or VisionDiagnoseSkill(
            text_skill=self.text_skill)
        self.generator = generator or AnswerGenerator(model=model)
        #generate / reject 复用 P0 原件，只替换 diagnose 用的 Skill
        self._diagnose, self._generate, self._reject = make_nodes(
            self.vision_skill, self.generator)
        self.model = model
        self._build()

    # ---------------------------------------------------------- 图构建
    def _build(self):
        def diagnose(state: VisionAgentState) -> dict:
            return self._diagnose(state)

        def generate(state: VisionAgentState) -> dict:
            return self._generate(state)

        def reject(state: VisionAgentState) -> dict:
            sk = state.get("skill_out", {})
            out = self._reject(state)          # 先跑 P0 的 reject 节点（拿 trace）
            if sk.get("route") == "vision":
                # 视觉拒答换文案：带上 top-2 与候选 KB
                out["answer"] = _format_vision_reject(sk)
            out["trace"][-1]["answer_kind"] = (
                "vision_reject" if sk.get("route") == "vision" else "text_reject")
            return out

        g = StateGraph(VisionAgentState)
        g.add_node("diagnose", diagnose)
        g.add_node("generate", generate)
        g.add_node("reject", reject)
        g.add_edge(START, "diagnose")
        g.add_conditional_edges("diagnose", route_after_diagnose,
                                {"generate": "generate", "reject": "reject"})
        g.add_edge("generate", END)
        g.add_edge("reject", END)
        self.graph = g.compile()

    # ---------------------------------------------------------- 对外接口
    def answer(self, query, image_paths=None, top_k=5, use_judge=True,
               expand=True, reject_threshold: float = DEFAULT_REJECT,
               device: str = "cuda", with_confusable: bool = True,
               model_dir=None):
        state = {"query": query, "top_k": top_k, "use_judge": use_judge,
                 "expand": expand, "trace": []}
        if image_paths:
            # 视觉参数挂在 state 上，由 diagnose 节点透传给 skill
            state["image_paths"] = list(image_paths)
            state["reject_threshold"] = reject_threshold
            state["device"] = device
            state["with_confusable"] = with_confusable
            state["model_dir"] = model_dir
        final = self.graph.invoke(state)

        sk = final.get("skill_out", {})
        return {
            "query": query,
            "route": sk.get("route", "text"),
            "rejected": final["rejected"],
            "layer": final["layer"],
            "reject_reason": final.get("reject_reason", ""),
            "answer": final["answer"],
            "kb_ids": final["kb_ids"],
            "evidence": sk.get("evidence", []),
            "top_cosine": final["top_cosine"],
            "vision": sk.get("vision", {}),
            "trace": final["trace"],
        }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ag = VisionDiagnoseAgent()
    out = ag.answer(a.query, image_paths=a.images, device=a.device)
    print(f"route={out['route']} rejected={out['rejected']} "
          f"layer={out['layer']} kb_ids={out['kb_ids']}")
    print("trace:", [(t["node"], t.get("route", "")) for t in out["trace"]])
    print(out["answer"][:400])


if __name__ == "__main__":
    main()
