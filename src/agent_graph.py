"""P2 编排层（主推实现）：LangGraph StateGraph 版 DiagnoseAgent。

与手写版（agent.py，保留作 baseline）逻辑等价，差异在编排方式：
手写版是顺序函数调用，本版把链路显式建成图——

    START → diagnose →(条件边)→ generate → END
                        └(拒答)→ reject   → END

图拓扑即文档：拒答分支、生成分支一目了然，且 LangGraph 自带
状态传递（State）与节点轨迹，后续加节点（如多轮澄清、Graph RAG 多跳）
只需加边，不用改调用栈。

设计要点
--------
1. **节点只做一件事**：diagnose 调 Skill，generate 调生成器，
   reject 产出拒答回复——与手写版复用同一套 Skill/Generator/缓存，
   两版行为可直接对比。
2. **trace 进 State**：每个节点把自己的关键输出写进 state["trace"]，
   评测脚本可审计全链路。
3. 条件边判据是 skill 输出的 rejected 标志（两层拒答已在 Skill 内完成）。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from skill_diagnose import DiagnoseSkill
from answer_generator import AnswerGenerator, ANSWER_MODEL
from agent import REJECT_REPLY


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    use_judge: bool
    expand: bool
    # diagnose 节点产出
    skill_out: dict
    rejected: bool
    # 终态产出
    answer: str
    reject_reason: str
    layer: str
    kb_ids: list
    top_cosine: float
    # reducer=operator.add：各节点返回的 trace 列表自动追加而非覆盖
    trace: Annotated[list, operator.add]


def _make_nodes(skill: DiagnoseSkill, generator: AnswerGenerator):
    def diagnose(state: AgentState) -> dict:
        sk = skill.run(state["query"],
                       top_k=state.get("top_k", 5),
                       use_judge=state.get("use_judge", True))
        return {
            "skill_out": sk,
            "rejected": sk["rejected"],
            "layer": sk["layer"],
            "kb_ids": sk["kb_ids"],
            "top_cosine": sk["top_cosine"],
            "trace": [{
                "node": "diagnose",
                "rejected": sk["rejected"],
                "layer": sk["layer"],
                "reason": sk["reason"],
                "kb_ids": sk["kb_ids"],
                "n_evidence": len(sk["evidence"]),
                "top_cosine": sk["top_cosine"],
            }],
        }

    def generate(state: AgentState) -> dict:
        sk = state["skill_out"]
        gen = generator.generate(state["query"], sk["evidence"],
                                 top_k=state.get("top_k", 5),
                                 expand=state.get("expand", True))
        return {
            "answer": gen["answer"],
            "reject_reason": "",
            "trace": [{
                "node": "generate",
                "cached": gen["cached"],
                "answer_chars": len(gen["answer"]),
            }],
        }

    def reject(state: AgentState) -> dict:
        sk = state["skill_out"]
        return {
            "answer": REJECT_REPLY,
            "reject_reason": sk["reason"],
            "trace": [{"node": "reject", "reason": sk["reason"]}],
        }

    return diagnose, generate, reject


def route_after_diagnose(state: AgentState) -> str:
    return "reject" if state["rejected"] else "generate"


def build_graph(skill=None, generator=None, model=ANSWER_MODEL):
    skill = skill or DiagnoseSkill()
    generator = generator or AnswerGenerator(model=model)
    diagnose, generate, reject = _make_nodes(skill, generator)

    g = StateGraph(AgentState)
    g.add_node("diagnose", diagnose)
    g.add_node("generate", generate)
    g.add_node("reject", reject)
    g.add_edge(START, "diagnose")
    g.add_conditional_edges("diagnose", route_after_diagnose,
                            {"generate": "generate", "reject": "reject"})
    g.add_edge("generate", END)
    g.add_edge("reject", END)
    return g.compile()


class GraphDiagnoseAgent:
    """对外接口与手写版 DiagnoseAgent.answer 对齐，便于双实现对比。"""

    def __init__(self, skill=None, generator=None, model=ANSWER_MODEL):
        self.graph = build_graph(skill, generator, model)

    def answer(self, query, top_k=5, use_judge=True, expand=True):
        final = self.graph.invoke({
            "query": query, "top_k": top_k,
            "use_judge": use_judge, "expand": expand,
            "trace": [],
        })
        # trace 字段声明了 reducer(operator.add)，各节点记录已自动累积
        return {
            "query": query,
            "rejected": final["rejected"],
            "layer": final["layer"],
            "reject_reason": final.get("reject_reason", ""),
            "answer": final["answer"],
            "kb_ids": final["kb_ids"],
            "top_cosine": final["top_cosine"],
            "trace": final["trace"],
        }


def main():
    agent = GraphDiagnoseAgent()
    for q in ["my print has stringing and oozing",
              "best filament brand to buy"]:
        out = agent.answer(q)
        print(f"\n=== {q} ===")
        print(f"rejected={out['rejected']} layer={out['layer']} "
              f"kb_ids={out['kb_ids']}")
        print("trace:", [s["node"] for s in out["trace"]])
        print(out["answer"][:200])


if __name__ == "__main__":
    main()
