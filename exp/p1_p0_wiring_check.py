"""P0↔P1 接线验收：`exp/p1_p0_wiring_check.py`

四道验收，覆盖「能用」与「没破坏」两类：

  A. **P0 零回归**（硬约束）—— 无图时必须与`DiagnoseSkill` 逐字段一致。
     ⚠️ 只比 `evidence` / `kb_ids` / `rejected` / `top_cosine`，**不比 skill 名**
     （本模块有意把 skill 改成 diagnose_vision 以便日志区分，且加了 route 字段）。
  B. **上下文预算**—— 5类映射各自的证据必须能原样通过 `build_context`
     而**不触发它的截断分支**（否则我们自己写的截断标注会被顶掉）。
  C. **视觉路径端到端** —— 真图 → 分类 → KB → evidence，与kb_mapping 对齐。
  D. **拒答透传** —— 低置信度图必须拒答，且 evidence 仍非空（供人工判断）。

跑法：`.venv/Scripts/python.exe -u exp/p1_p0_wiring_check.py`
结果落`exp/p1_wiring_result.json`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import json as _json                                  # noqa: E402
from answer_generator import build_context, MAX_CONTEXT_CHARS   # noqa: E402
from skill_diagnose import DiagnoseSkill             # noqa: E402
from skill_vision_diagnose import (                  # noqa: E402
    VisionDiagnoseSkill, EVIDENCE_BUDGET_CHARS, TOP_COSINE_NA,
)
from src.vision.kb_mapping import build_mapping, load_kb   # noqa: E402
from src.vision.pipeline import DEFAULT_REJECT              # noqa: E402

PICK = ROOT / "exp" / "_wiring_pick.json"
OUT = ROOT / "exp" / "p1_wiring_result.json"

#: 验收 A 用的文本 query（第一条是库内正例，第二条是库外负例）
TEXT_QUERIES = [
    "my print has stringing and oozing",
    "best filament brand to buy",
]


def assert_no_bypass_needed() -> str:
    """确认 P0 既有 bug 已真修：**不再需要任何进程内旁路**。

    历史：本脚本原先有 `patch_offline_embedder()`，靠在验收进程内预置
    模块级 `_FITTED_EMBEDDER` 来绕过 P0 的
    `vector_index.load_fitted_embedder()` 硬编码路径 bug。
    该bug 已于 2026-10-05 获用户批准后**真修**：
      - `load_fitted_embedder(path=None, strict=False)` 按索引路径取；
      - `_FITTED_EMBEDDER_CACHE` 按索引路径分桶；
      - `HybridRetriever.faiss_path` 透传到 `_embed_query`；
      - `strict=True` 时文件缺失直接抛 `FileNotFoundError`。
    所以旁路已删除；这里反向自检——**若修复被回退，本函数必须报错**。
    """
    import hybrid_retriever as hr
    from config import KB_FAISS_INDEX_PATH

    if not hasattr(hr, "_FITTED_EMBEDDER_CACHE"):
        raise AssertionError(
            "hr._FITTED_EMBEDDER_CACHE 不存在 → P0 修复被回退，"
            "或代码跑在旧版本上。"
        )
    if hasattr(hr, "_FITTED_EMBEDDER"):
        raise AssertionError(
            "旧的全局面变量 _FITTED_EMBEDDER 仍在 → P0 修复未生效。"
        )
    p = KB_FAISS_INDEX_PATH.with_suffix(".tfidf_svd.pkl")
    if not p.exists():
        return f"跳过 fitted 文件检查（无 {p.name}）"
    # 直接调 strict=True：修复前会因路径硬编码而找不到文件（strict 抛错）
    import vector_index as vi
    fitted = vi.load_fitted_embedder(KB_FAISS_INDEX_PATH, strict=True)
    vocab = len(fitted[0].vocabulary_)
    if vocab < 1000:
        raise AssertionError(
            f"取到的是迷你模型（vocab={vocab}），不是 kb 级 fitted（应为万级）。"
        )
    # 真跑一次纯 P0 检索，确认不经任何旁路也能通
    r = hr.HybridRetriever()
    r.retrieve(TEXT_QUERIES[0], top_k=3)
    return (f"无需旁路：按路径取到 kb 级 fitted（vocab={vocab}），"
            f"纯 P0 检索通路正常")


def check_a_p0_zero_regression(vskill: VisionDiagnoseSkill) -> dict:
    """A. 无图路径必须与 P0 逐字段一致。"""
    base = DiagnoseSkill()
    rows, ok = [], True
    for q in TEXT_QUERIES:
        # 🔴 use_judge=False：A 项只验「文本路径是否被污染」，
        # 不验判官（判官要调 dashscope，本机无 key，且与接线无关）。
        a = base.run(q, use_judge=False)       # P0 原件
        b = vskill.run(q, image_paths=None, use_judge=False)  # 视觉 skill 文本分支
        cmp = {}
        for k in ("rejected", "layer", "reason", "kb_ids", "top_cosine"):
            same = a[k] == b[k]
            cmp[k] = same
            ok &= same
        # evidence 逐块比（文本、顺序、字段）
        ev_same = ([(e["chunk_id"], e["kb_id"], e["problem"], e["text"])
                    for e in a["evidence"]]
                   == [(e["chunk_id"], e["kb_id"], e["problem"], e["text"])
                       for e in b["evidence"]])
        cmp["evidence(逐块含顺序)"] = ev_same
        ok &= ev_same
        rows.append({"query": q, "fields": cmp,
                     "kb_ids": a["kb_ids"], "n_evidence": len(a["evidence"])})
    return {"pass": ok, "note": ("无图 → 委托 DiagnoseSkill，"
                                 "evidence/kb_ids/rejected/layer/reason/"
                                 "top_cosine 全部逐字段一致"),
            "queries": rows}


def check_b_context_budget(vskill: VisionDiagnoseSkill) -> dict:
    """B. 5 类映射的证据都必须不触发 build_context 的二次截断。"""
    mp = build_mapping(load_kb())
    kb2 = {e["kb_id"]: e for e in
           _json.loads((ROOT / "data" / "kb_v2.json").read_text(encoding="utf-8"))}
    rows, ok = [], True
    for cls, m in mp.items():
        ids = [k.kb_id for k in m.primary + m.secondary + m.confusable]
        ev, used, trunc = vskill._build_evidence(ids)
        ctx = build_context(ev)
        retriggered = "预算用尽" in ctx
        all_kept = len(used) == len(ids)
        within = len(ctx) <= MAX_CONTEXT_CHARS
        shown = sum(e["n_causes_shown"] for e in ev)
        total = sum(len(kb2[e["kb_id"]]["causes"]) for e in ev)
        ok &= (not retriggered) and all_kept and within
        rows.append({"cls": cls, "n_kb": len(ids), "n_kb_used": len(used),
                     "ctx_chars": len(ctx), "budget": MAX_CONTEXT_CHARS,
                     "build_context_retriggered": retriggered,
                     "all_kb_kept": all_kept, "within_budget": within,
                     "causes_shown": shown, "causes_total": total,
                     "coverage": round(shown / max(total, 1), 4),
                     "truncated": trunc})
    return {"pass": ok, "note": ("每类映射的证据都完整保留所有 KB、"
                                 "不触发 build_context 二次截断、"
                                 "且总长 ≤ P0 硬编码的 10000"),
            "per_class": rows}


def check_c_vision_e2e(vskill: VisionDiagnoseSkill, device="cuda") -> dict:
    """C. 真图 → 分类 → KB → evidence，与 kb_mapping 对齐。"""
    if not PICK.exists():
        return {"pass": False, "error": f"缺少 {PICK}，先抽样图片"}
    picks = _json.loads(PICK.read_text(encoding="utf-8"))
    mp = build_mapping(load_kb())

    # 一次跑完5 张（省一次模型加载）
    from src.vision.pipeline import diagnose as vdiag
    preds = vdiag([p["path"] for p in picks], device=device)

    rows, ok = [], True
    for pick, pr in zip(picks, preds):
        m = mp[pr.pred_class]
        expect = list(dict.fromkeys(k.kb_id
                                    for k in m.primary + m.secondary + m.confusable))
        got = pr.kb_ids
        match = got == expect
        ok &= match and not pr.rejected
        rows.append({"true_label": pick["label"], "pred_class": pr.pred_class,
                     "correct": pick["label"] == pr.pred_class,
                     "proba": round(pr.proba, 4),
                     "rejected": pr.rejected, "kb_ids": got,
                     "kb_ids_match_mapping": match,
                     "n_causes_shown": {e["kb_id"]: e["n_causes_shown"]
                                        for e in vskill._build_evidence(got)[0]}})
    n_correct = sum(r["correct"] for r in rows)
    return {"pass": ok, "n_images": len(rows),
            "n_correct": n_correct,
            "note": ("每张图的 kb_ids 必须与 kb_mapping 全量映射一致"
                     "（primary→secondary→confusable 去重保序）"),
            "rows": rows}


def check_d_reject_passthrough(vskill: VisionDiagnoseSkill,
                               device="cuda") -> dict:
    """D. 阈值调到1.0 → 必然拒答，且 evidence 仍非空（top-2 候选留给人工）。"""
    picks = _json.loads(PICK.read_text(encoding="utf-8")) if PICK.exists() else []
    if not picks:
        return {"pass": False, "error": "无抽样图片"}
    out = vskill.run("这张图是什么缺陷？", image_paths=[picks[0]["path"]],
                     reject_threshold=1.01, use_judge=False, device=device)
    ok = (out["rejected"] and out["layer"] == "vision_confidence"
          and len(out["evidence"]) > 0
          and all(e["source"] == "vision" for e in out["evidence"]))
    return {"pass": ok,
            "note": ("视觉拒答时 evidence 仍非空（pipeline 的「拒答仍输出 top-2」"
                     "纪律在Skill 层同样成立），layer=vision_confidence"),
            "rejected": out["rejected"], "layer": out["layer"],
            "reason": out["reason"], "n_evidence": len(out["evidence"]),
            "kb_ids": out["kb_ids"]}


def check_e_contract(vskill: VisionDiagnoseSkill) -> dict:
    """E. 契约一致性：两条路径都提供 P0 下游依赖的全部字段。

    🔴 `top_cosine` 的取值**按路径不同**，不是恒为 -1.0：
      文本路径 = 真实检索 top-1 余弦（如 0.3355）；
      视觉路径 = -1.0（无余弦相似度这个量）。
    早先版本在这里断言「恒等于 -1.0」→ 文本路径必然 FAIL。
    教训同项目记忆里那条：**验收断言要先问「这个量在两条路径上语义相同吗」**。
    """
    need = {"skill", "rejected", "layer", "reason", "evidence",
            "kb_ids", "top_cosine"}
    text_out = vskill.run("test", image_paths=None, use_judge=False)
    missing_text = sorted(need - set(text_out.keys()))
    # 视觉路径：真图（复用抽样第一张），不跑判官以免依赖 dashscope
    picks = _json.loads(PICK.read_text(encoding="utf-8")) if PICK.exists() else []
    if not picks:
        return {"pass": False, "error": "无抽样图片，无法验视觉路径契约"}
    vis_out = vskill.run("这张图是什么缺陷？", image_paths=[picks[0]["path"]],
                         use_judge=False, device="cpu")
    missing_vis = sorted(need - set(vis_out.keys()))

    ok = (not missing_text and not missing_vis
          and vis_out["top_cosine"] == TOP_COSINE_NA
          and vis_out["route"] == "vision"
          and text_out["route"] == "text")
    return {"pass": ok,
            "missing_text_route": missing_text, "missing_vision_route": missing_vis,
            "top_cosine_text_route": text_out["top_cosine"],
            "top_cosine_vision_route": vis_out["top_cosine"],
            "note": ("两路径字段均齐备；top_cosine 按路径取值"
                     "（文本=真实余弦 / 视觉=-1.0 表示不适用，"
                     "复用 P0 无结果时的取值而非 0/1）")}

def check_f_graph_routing(device="cuda") -> dict:
    """F. 编排层图路由：文本零回归 + 图像真走视觉 + 多图不混类别。

    🔴 这项抓过一个**静默失效**：LangGraph 对schema 未声明的入参
    **直接丢弃且不报错**——`image_paths` 传进去就没了，有图也走文本路径，
    而所有返回值都「看起来正常」。详见 agent_vision.VisionAgentState 注释。
    所以这里必须断言 `route == "vision"`，不能只看字段齐不齐。
    """
    from agent_vision import VisionDiagnoseAgent
    from agent_graph import GraphDiagnoseAgent

    class StubGen:
        """桩生成器：本项只验路由与 evidence 传递，不烧 API。"""
        def __init__(self):
            self.calls = []

        def generate_with_backfill(self, query, evidence, **kw):
            self.calls.append([e["kb_id"] for e in evidence])
            return {"answer": f"STUB n_ev={len(evidence)}",
                    "cached": False, "backfilled": False}

    g_p0, g_vis = StubGen(), StubGen()
    p0 = GraphDiagnoseAgent(generator=g_p0)
    vis = VisionDiagnoseAgent(generator=g_vis)

    # ---- F1 文本路径逐字段零回归 ----
    q = TEXT_QUERIES[0]
    a = p0.answer(q, use_judge=False)
    b = vis.answer(q, use_judge=False)
    same = {k: a[k] == b[k] for k in
            ("rejected", "layer", "answer", "kb_ids", "top_cosine")}
    same["evidence序列"] = g_p0.calls == g_vis.calls
    ok_text = all(same.values())

    # ---- F2 每张图都必须走 vision 路由 ----
    picks = _json.loads(PICK.read_text(encoding="utf-8")) if PICK.exists() else []
    routes, route_ok = [], True
    for p in picks:
        out = vis.answer("这图是什么缺陷？", image_paths=[p["path"]],
                         use_judge=False, device=device)
        routes.append({"true": p["label"], "route": out["route"],
                       "pred": out["vision"]["pred_class"],
                       "proba": out["vision"]["proba"],
                       "kb_ids": out["kb_ids"]})
        route_ok &= (out["route"] == "vision"
                     and out["vision"]["pred_class"] == p["label"])

    # ---- F3 多图：类别互斥，绝不能把多类 KB 混进同一份 evidence ----
    multi_ok, multi = True, {}
    if len(picks) >= 2:
        out = vis.answer("这图是什么缺陷？",
                         image_paths=[p["path"] for p in picks[:3]],
                         use_judge=False, device=device)
        v = out["vision"]
        expect = list(v["kb_ids_used"])
        # 只含主类别的映射，不含其他类别的 KB
        single = vis.answer("x", image_paths=[picks[0]["path"]],
                            use_judge=False, device=device)["kb_ids"]
        mixed = len(v["class_conflict"]) > 1
        multi_ok = (out["kb_ids"] == expect
                    and (v["kb_ids"] == single if not mixed else True)
                    and (v["conflict_note"] != "" if mixed else True))
        multi = {"n_images": v["n_images"], "pred": v["pred_class"],
                 "class_conflict": v["class_conflict"],
                 "kb_ids": out["kb_ids"], "n_kb": len(out["kb_ids"]),
                 "conflict_note": v["conflict_note"]}

    return {"pass": ok_text and route_ok and multi_ok,
            "text_route_zero_regression": same,
            "per_image_routes": routes,
            "multi_image": multi,
            "note": ("文本路径与 P0 逐字段一致；每张图 route=vision；"
                     "多图只取主类别映射 +显式告知类别冲突")}


def main():
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        pass

    vskill = VisionDiagnoseSkill()
    print(f"[device] {device}")
    # 🔴 反向自检：P0 bug 已真修，本脚本不再做任何进程内旁路。
    #    若修复被回退，这里会直接 AssertionError 而不是静默跑偏。
    print(f"[P0 修复自检] {assert_no_bypass_needed()}\n")

    results = {}
    for name, fn in [("A_p0_zero_regression", lambda: check_a_p0_zero_regression(vskill)),
                     ("B_context_budget", lambda: check_b_context_budget(vskill)),
                     ("E_contract", lambda: check_e_contract(vskill)),
                     ("C_vision_e2e", lambda: check_c_vision_e2e(vskill, device)),
                     ("D_reject_passthrough", lambda: check_d_reject_passthrough(vskill, device)),
                     ("F_graph_routing", lambda: check_f_graph_routing(device))]:
        print(f"── {name} ──")
        try:
            r = fn()
        except Exception as e:                       # noqa: BLE001
            r = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        results[name] = r
        print(("  PASS  " if r.get("pass") else "  FAIL  ")
              + str(r.get("note", r.get("error", "")))[:110])
        for key in ("per_class", "queries", "rows", "per_image_routes"):
            for row in r.get(key, []):
                print("   ", _json.dumps(row, ensure_ascii=False)[:190])
        if "multi_image" in r:
            print("   ", _json.dumps(r["multi_image"], ensure_ascii=False)[:250])
        if "text_route_zero_regression" in r:
            print("    文本零回归:", r["text_route_zero_regression"])
        if "rejected" in r:
            print("   ", {k: r[k] for k in ("rejected", "layer", "n_evidence")})
        print()

    all_pass = all(r.get("pass") for r in results.values())
    OUT.write_text(_json.dumps(
        {"all_pass": all_pass, "device": device,
         "budget": EVIDENCE_BUDGET_CHARS,
         "p0_max_context": MAX_CONTEXT_CHARS,
         "default_reject_threshold": DEFAULT_REJECT,
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("ALL PASS" if all_pass else "有验收项未通过")
    print(f"[产出] {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
