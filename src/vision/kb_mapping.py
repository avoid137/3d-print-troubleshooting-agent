"""P1 视觉类 → 知识库条目映射层（唯一权威源）。

为什么用代码而不是 md（2026-10-04 决定）：
    旧的 `data/mapping.md` 基于已弃用的 8 类体系，且它引用的 45 个 KB ID 中有
    **8 个在 kb_v2.json 中不存在**（KB029/030/045/059/066/068/070/073，只在
    kb_v1 的 75 条里）。若下游 import 一个写错的 ID 列表会静默失真。
    故本模块从 `data/kb_v2.json`（56 条，唯一在用版本）直接读，构造映射，
    并在导入时校验每个 KB ID 真实存在 —— 不存在的直接抛错，不静默通过。

映射设计要点：
    · 3 类同名可直接映射（Stringing / Warping / Layer_shifting）
    · Cracking / Off_platform 跨体系，须逐条核对知识库后再定
    · 每条映射显式记录 `confusable_kb` 与 `conflict_reason`：
      把「看起来像但不是」的知识库条目写清楚，避免下游把不同失效模式混为一谈。
      （旧mapping.md 曾把 Spaghetti 与 Stringing 的成因混淆，这是同类错误。）

⚠️ 待人工核对（数据到位后必须做，勿跳过）：
    `pending_visual_check = True` 的条目，其`primary_kb` 是**基于现象描述的推断**，
    必须看图确认后才算定稿。报告中须写明「看了几张图、结论是什么」。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

KB_PATH = Path("data/kb_v2.json")
KB_REL = "F:/workbuddy/3d-print-troubleshooting-agent/data/kb_v2.json"


# ---------------------------------------------------------------- 数据结构
@dataclass
class KBLink:
    kb_id: str
    title: str
    cause_count: int
    #: direct=名称直接对应 / semantic=语义对应需人工确认 / anchor=kb_v2 中已标注 img_classes 的现成锚点
    match: str
    role: str = "primary"          # primary=主映射 / secondary=次要 / confusable=易混淆（非同一现象）
    note: str = ""


@dataclass
class VisualMapping:
    visual_class: str
    zh: str
    n_images: int
    primary: list = field(default_factory=list)
    secondary: list = field(default_factory=list)
    confusable: list = field(default_factory=list)
    conflict_reason: str = ""
    pending_visual_check: bool = True
    checked_images: int = 0
    check_note: str = ""

    def all_kb_ids(self, include_confusable: bool = False) -> list:
        ids = [k.kb_id for k in self.primary + self.secondary]
        if include_confusable:
            ids += [k.kb_id for k in self.confusable]
        return ids


# ---------------------------------------------------------------- 读 KB
def load_kb(path=None) -> dict:
    p = Path(path) if path else (KB_PATH if KB_PATH.exists() else KB_REL)
    kb = json.loads(p.read_text(encoding="utf-8"))
    return {e["kb_id"]: e for e in kb}


# ---------------------------------------------------------------- 映射定义
def build_mapping(kb: dict) -> dict:
    def L(kb_id, match, role="primary", note=""):
        e = kb.get(kb_id)
        if e is None:
            raise KeyError(f"KB{kb_id} 不在 kb_v2.json 中 —— 映射层拒绝引用不存在的条目")
        return KBLink(kb_id=kb_id, title=e["problem"],
                      cause_count=e["cause_count"], match=match,
                      role=role, note=note)

    m: dict[str, VisualMapping] = {}

    # ---------- 1) Stringing（同名，kb_v2 已有 img_classes 锚点）----------
    m["Stringing"] = VisualMapping(
        visual_class="Stringing", zh="拉丝/渗料", n_images=447,
        primary=[L("KB006", "anchor", note="kb_v2 已标注 img_classes=['Stringing']"),
                 L("KB063", "direct", note="标题即 Stringing，现象更具体（覆盖全件的网状丝）")],
        confusable=[L("KB101", "semantic", "confusable",
                      "⚠️ Spaghetti 是喷不出料成团丝，与拉丝不是同一现象")],
        conflict_reason="Stringing（回抽不足产生的细丝）与 Spaghetti（完全不出料/成团）"
                        "成因不同，不可混用。二者在 kb_v2 中均为 FDM 条目但机理相反。",
        pending_visual_check=False,
        check_note="3 类同名映射，kb_v2 已有 20/56 条 img_classes 锚点可依，无需看图定稿")

    # ---------- 2) Warping（同名）----------
    m["Warping"] = VisualMapping(
        visual_class="Warping", zh="翘曲", n_images=538,
        primary=[L("KB024", "anchor", note="kb_v2 img_classes=['Warping']，11 条根因，条目最完整"),
                 L("KB044", "direct", note="Elephant's Foot（底部外扩），Warping 的特例"),
                 L("KB016", "direct", note="翘边/圆角，边缘翘起的表现")],
        confusable=[],
        conflict_reason="",
        pending_visual_check=False,
        check_note="3 类同名映射，kb_v2 已有 img_classes 锚点")

    # ---------- 3) Layer_shifting（同名）----------
    m["Layer_shifting"] = VisualMapping(
        visual_class="Layer_shifting", zh="层错位", n_images=364,
        primary=[L("KB008", "anchor", note="kb_v2 img_classes=['Layer Shifting']，7 条根因"),
                 L("KB050", "direct", note="层对不齐，程度更轻的表现"),
                 L("KB052", "direct", note="整体倾斜（打印件歪倒）"),
                 L("KB065", "direct", note="局部偏移")],
        confusable=[L("KB009", "semantic", "confusable",
                      "⚠️ 层分离/开裂是层间结合失效，与层错位（几何错位）不同")],
        conflict_reason="Layer_shifting 是**几何位置错位**；KB009/KB049 是**层间结合失效**"
                        "（Delamination）。前者应调机械/运动参数，后者要调温度与层间粘合。",
        pending_visual_check=False,
        check_note="3 类同名映射，kb_v2 已有 img_classes 锚点")

    # ---------- 4) Cracking（⚠️ 跨体系，必须看图定稿）----------
    m["Cracking"] = VisualMapping(
        visual_class="Cracking", zh="开裂", n_images=472,
        primary=[L("KB049", "semantic", "primary",
                   "唯一以「裂纹/cracks」为现象主语的条目；推断 DC-02R 的 Cracking 即此"),
                 ],
        secondary=[],
        confusable=[L("KB009", "semantic", "confusable",
                      "⚠️ 关键区分：KB009 是「层分离与劈裂（Delamination）」，"
                      "KB049 是「高大件上出现裂纹」。**这是两种失效模式**，"
                      "kb_v2 把两者都标了 img_classes=['Delamination']，极易混淆")],
        conflict_reason="Cracking 与 Delamination 在 kb_v2 的 img_classes 里同标签，"
                        "但物理机理不同：裂纹=材料内部应力破坏；层分离=层间粘合失效。"
                        "若硬映射到 KB009，会把「调温度/粘合」的建议给到「调应力/结构」的场景。",
        pending_visual_check=True,
        checked_images=0,
        check_note="TODO：下载后抽 15~20 张 Cracking 图人工看，确认是表面/内部裂纹"
                   "（→KB049）还是层间分离（→KB009），再定稿。禁止凭想象定稿。")

    # ---------- 5) Off_platform（⚠️ 跨体系，必须看图定稿）----------
    m["Off_platform"] = VisualMapping(
        visual_class="Off_platform", zh="脱离平台", n_images=91,
        primary=[L("KB002", "semantic", "primary",
                   "18 条根因，覆盖「不粘床」全部成因（温度/清洁/调平/Z-offset/材料）"),
                 L("KB028", "semantic", "secondary",
                   "首层问题 13 条根因，与「脱离平台」常同源（首层没粘住→后续翘起脱离）")],
        secondary=[L("KB035", "weak", "secondary", "喷嘴离床过近（2 条根因，属特定成因"),
                     L("KB037", "weak", "secondary", "打印头未覆盖到床（3 条根因）")],
        confusable=[L("KB101", "semantic", "confusable",
                      "⚠️ KB101 标题含 'Print Detached from Bed'，字面像本类，"
                      "但其机制是 Spaghetti（喷不出料成团丝）——**标题相近、机理不同**，"
                      "旧 mapping.md 正是在这里把 Spaghetti 与 Stringing 混淆过"),
                    L("KB067", "semantic", "confusable",
                      "反向现象：粘在床上下不来，与本类相反")],
        conflict_reason="Off_platform 是「打印件脱离打印床」；需区分两种可能："
                        "(a) 首层未粘附（→KB002/KB028，附着力问题）；"
                        "(b) 打印中途翘起脱落（→翘曲 + 附着力，跨 Warping 与本类）。"
                        "Spaghetti 虽然标题含 'Detached from Bed'，但机制不同，不可混用。",
        pending_visual_check=True,
        checked_images=0,
        check_note="TODO：Off_platform 仅 91 张，建议全部人工过一遍，"
                   "判定是「首层没粘住」还是「打印中翘起脱落」，再决定主映射权重。")
    return m


# ---------------------------------------------------------------- 校验
def validate(m: dict, kb: dict) -> None:
    bad = []
    for vc, mp in m.items():
        for k in mp.primary + mp.secondary + mp.confusable:
            if k.kb_id not in kb:
                bad.append(f"{vc} → {k.kb_id}")
    if bad:
        raise KeyError(f"映射引用了不存在的 KB 条目: {bad}")

    pend = [vc for vc, mp in m.items() if mp.pending_visual_check]
    print(f"[mapping] 5类映射构建完成，共 {len(m)} 类")
    print(f"[mapping] KB 引用校验通过（全部存在于 kb_v2.json {len(kb)} 条）")
    print(f"[mapping] ⚠️ 待看图定稿: {pend}" if pend else "[mapping] 全部已定稿")
    cov = len(m) / 5
    print(f"[mapping] 覆盖率: {len(m)}/5 = {cov:.0%}（验收要求 100% 覆盖）")


def to_markdown(m: dict) -> str:
    L = ["| 视觉类 | 张数 | 主映射 | 次要 | 易混淆（**非同一现象**） | 看图定稿 |",
         "|---|---|---|---|---|---|"]
    for vc, mp in m.items():
        f = lambda ks: "<br>".join(f"`{k.kb_id}` {k.title}" for k in ks) or "—"
        L.append(f"| **{mp.visual_class}**<br>{mp.zh} | {mp.n_images} | {f(mp.primary)} "
                 f"| {f(mp.secondary)} | {f(mp.confusable)} "
                 f"| {'❌ 待做' if mp.pending_visual_check else '✅ 已定'} |")
    return "\n".join(L)


def main():
    kb = load_kb()
    m = build_mapping(kb)
    validate(m, kb)
    print()
    print(to_markdown(m))
    print()
    for vc, mp in m.items():
        if mp.conflict_reason:
            print(f"[冲突说明] {vc}：{mp.conflict_reason}\n")
    out = Path("exp/p1_kb_mapping.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {k: asdict(v) for k, v in m.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[产出] {out}")


if __name__ == "__main__":
    main()
