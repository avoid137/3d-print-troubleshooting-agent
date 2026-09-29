"""P2 Agent 编排层：路由 → Skill 调用 → 上下文管理 → 答案生成 / 拒答回复。

这是 P2 的核心：不再是「检索返回片段」，而是端到端产出
**最终诊断答案或得体的拒答**，并记录全链路轨迹（trace）供评测与调试。

链路
----
    query
      → DiagnoseSkill（检索 + 两层拒答）
      → 拒答 → 生成拒答回复（模板，零 API，说明知识库覆盖范围）
      → 放行 → AnswerGenerator（证据全文 → 结构化诊断答案）
      → 返回 trace（每一步的输入输出与耗时）

上下文管理（harness 职责）体现在两处：
    1. Skill 层的 parent-child 扩展（检索 cause 级、补齐同 problem 兄弟块）；
    2. 生成层的上下文预算裁剪（见 answer_generator.build_context）。
"""
import time

from skill_diagnose import DiagnoseSkill
from answer_generator import AnswerGenerator, ANSWER_MODEL

REJECT_REPLY = (
    "抱歉，这个问题超出了本知识库的覆盖范围。\n"
    "本知识库是一份 FDM 打印质量缺陷排障手册，覆盖：挤出/附着/表面/结构/几何"
    "类缺陷的根因与解决方案、相关打印参数与校准（温度、回抽、流量、层高、速度、"
    "风扇、Z 缝等）、以及这些缺陷的机械类根因（背隙、皮带张紧、微步、丝杆定位）。\n"
    "不覆盖：设备选型与硬件改装、接线与固件、切片/CAD 软件操作教程、"
    "材料选购与存储、非 FDM 工艺（SLA/DLP/树脂）、工艺科普、商业运营、"
    "后处理与上色。\n"
    "如果你的问题是关于打印质量缺陷的，欢迎换一种方式描述现象再来问我。"
)


class DiagnoseAgent:
    def __init__(self, skill=None, generator=None, model=ANSWER_MODEL):
        self.skill = skill or DiagnoseSkill()
        self.generator = generator or AnswerGenerator(model=model)

    def answer(self, query, top_k=5, use_judge=True, expand=True):
        """端到端回答。返回含 trace 的完整结构。"""
        t0 = time.time()
        trace = {"query": query, "steps": []}

        # Step 1: Skill（检索 + 拒答）
        sk = self.skill.run(query, top_k=top_k, use_judge=use_judge)
        trace["steps"].append({
            "skill": sk["skill"],
            "rejected": sk["rejected"],
            "layer": sk["layer"],
            "reason": sk["reason"],
            "kb_ids": sk["kb_ids"],
            "n_evidence": len(sk["evidence"]),
            "top_cosine": sk["top_cosine"],
        })

        if sk["rejected"]:
            out = {
                "query": query,
                "rejected": True,
                "layer": sk["layer"],
                "reject_reason": sk["reason"],
                "answer": REJECT_REPLY,
                "kb_ids": [],
                "top_cosine": sk["top_cosine"],
                "elapsed_s": round(time.time() - t0, 2),
                "trace": trace,
            }
            return out

        # Step 2: 答案生成
        gen = self.generator.generate(query, sk["evidence"],
                                      top_k=top_k, expand=expand)
        trace["steps"].append({
            "generator": "answer",
            "cached": gen["cached"],
            "answer_chars": len(gen["answer"]),
        })

        return {
            "query": query,
            "rejected": False,
            "layer": sk["layer"],
            "reject_reason": "",
            "answer": gen["answer"],
            "kb_ids": sk["kb_ids"],
            "top_cosine": sk["top_cosine"],
            "elapsed_s": round(time.time() - t0, 2),
            "trace": trace,
        }


def main():
    agent = DiagnoseAgent()
    for q in ["my print has stringing and oozing",
              "best filament brand to buy"]:
        out = agent.answer(q)
        print(f"\n=== {q} ===")
        print(f"rejected={out['rejected']} layer={out['layer']} "
              f"kb_ids={out['kb_ids']} ({out['elapsed_s']}s)")
        print(out["answer"][:300])


if __name__ == "__main__":
    main()
