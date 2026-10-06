"""P1 · 推理侧：把 L2 线性探针落盘成可复用的「部署件」。

🔴 为什么必须有这个模块（2026-10-05 发现）：
    `train.py` 跑完只写 json，**模型完全没有落盘**。
    这意味着下游要预测一张新图，只能重跑 5 折 × 2 种子训练（约 4 分钟），
    「缺陷图 → KB 条目」的端到端链路根本串不起来。
    本模块负责：全量重训 → 落盘（含 scaler/clf/骨干权重/配置/指纹）→ 加载预测。

🔴🔴 最重要的一条纪律：**部署件不能报 99.73%**
    `train.py` 的 99.73% 来自 **group-aware 5 折交叉验证**：
    每张图的预测都来自「没见过这张图」的模型，是泛化估计。
    而本模块落盘的模型是**在全量 1,912 张上重训**的（这是部署所需——
    交叉验证版本每张只用了 80% 数据，浪费了 20%）。
    在全量模型上重新评估同一批图 = **训练集准确率**，必然虚高。
    故：
      · 落盘时记录 `eval_note` 明确标注「此模型未做泛化评估」
      · `evaluate()` 函数**拒绝**在全量模型上算 macro-F1 作为性能指标
      · 性能数字只允许引用 `train.py` 的折外结果，通过 `load_deployed_model`
        的 `expected_cv_macro_f1` 字段做**一致性对��**，不重算
    这与 P0 阶段「模型选择曾误用测试折」的 bug 是同一类错误，必须防复发。

落盘内容（单目录 `data/_model/`）：
    l2_deploy.joblib   scaler + clf + classes + 元信息
    l2_deploy.json     人可读的元信息（对账用）
    backbone.safetensors
        骨干权重（从 HF cache 复制，不重新下载）
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

#: 视觉缺陷 5 类。**顺序即部署件里的列索引，改动会让旧部署件失效**——
#: load_deployed_model() 会断言 blob["classes"] == CLASSES，不一致直接抛错。
DEFECT_CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting",
                   "Off_platform"]
#: 第 6 类：无缺陷/正常件。2026-10-06 起加入（见下）。
#:
#: 🔴 **为什么必须加**：此前是 5 类**强制分类**，任何输入都必须被归进5 个缺陷类之一
#:   → 好件必然被误报。D8 域外实测：正常件误报率 **72%~75%**，
#:   且 86.6% 的图（含 91.0% 的正常件）被统一判成 Stringing。
#:   这不是调阈值能解决的——把拒答阈值从 0.60 调到 0.95，
#:   正常件漏拒率只从 91% 降到 62%。
#:
#: ✅ **加类后实测**（exp/d9_ab_normal_class.py，A/B 对照，group-aware 折外）：
#:   DC-04 正常件正确识别 **0% → 100%**；DC-02R 5 类 macro-F1 仅 **−0.08pp**
#:   （N=200）。**这是「误报抑制」，不是「泛化修复」**——DC-04 缺陷件之间的
#:   区分仍然塌缩（预测最大类占比 99.4%），因为跨机型/相机的特征漂移
#:   是冻结骨干+线性探针无法解决的（该方案假设「测试分布 = 训练分布」）。
NORMAL_CLASS = "No_error"
CLASSES = DEFECT_CLASSES + [NORMAL_CLASS]

#: DC-04 正常件（Zenodo 18628777，CC BY 4.0）在部署件里的用量。
#: N=50/100/200 三档实测：N≥100 时正常件识别已 100%。
#:
#: 🔴🔴 **2026-10-06 落地实测：DC-04 正常件不能作为通用正常类样本。**
#:    端到端回归发现 **DC-04 的缺陷件也有 100% 被判成 No_error**（p≈0.999）。
#:    根因（已测特征空间，非推测）：
#:      · DC-04 三类质心与 DC-02R 全体质心余弦 = 0.53~0.55，**三类几乎重合**；
#:      · 正常件质心相对缺陷类质心只领先 +0.019，`balanced` 再放大 1.76×；
#:      → 模型学到的是「**像 DC-04** ⇒ 正常」，不是「**无缺陷** ⇒ 正常」。
#:    `class_weight=None` 对此**毫无改善**（同样 100%）→ 不是权重问题，
#:    是特征空间本身不可分。**调阈值也救不了**（塌缩下高置信度是危险信号）。
#:
#:    → 因此**默认 0**：不把 DC-04 正常件混进部署件。
#:      要开启必须显式 `--n-normal N`，且**会被落盘前护栏拦截**
#:      （`_sanity_check_ood`，已反向自检确认真的会 FAIL）。
#:    → 正确解法只有一个：**自采与目标域同分布的正常件**（换机型/换材料必须重采）。
#:
#:    ⚠️ D9 的结论**没有错**，但边界被划窄了：
#:    它测的是「DC-02R 缺陷 + DC-04 正常」这一混合分布，
#:    **从未验证「域外缺陷件不会被误判为正常」**——而这恰是上线后最致命的一项。
#:    少测的那一项现在由护栏强制拦截。
N_NORMAL_TRAIN = 0
#: DC-04 zip 路径（正常件来源）。原始数据不入 git，这里只存路径。
DC04_ZIP = Path(r"D:/datasets/dc04/test_data.zip")

MODEL_DIR = ROOT / "data" / "_model"
IMG = 224
BACKBONE = "efficientnet_b0"
#: train.py 折外实测值（L2 / B0 / group-aware5 折 × 2 种子）
CV_MACRO_F1 = 0.9973
#: 🔴 2026-10-06：**部署件默认不混正常件**（`NORMAL_TRAIN = 0`），
#:   所以 DC-02R 的 5 类折外 macro-F1 就是历史值 **99.73%**，零损伤。
#:
#:   ⚠️ 曾有一版口径写成「6 类 / balanced 下 DC-02R 5 类 99.73%、零损伤」
#:   （exp/d9_class_weight_check.json）——**那个数字只在折外成立，不能搬进部署件**：
#:   折外时 DC-04 正常件被held-out，而**部署件是全量重训**，
#:   正常件被反复见到 → 类别吞噬严重得多（实测 100% 缺陷件被判正常）。
#:   **折外结论不能直接搬到全量部署件**，这是「部署件高置信度是自证」的另一面。
CV_MACRO_F1_DEFECT_5 = 0.9973
#: 🔴 「DC-04 正常件折外识别率 100%」**仅在折外成立，且不代表跨域可用**
#:   ——部署件上 DC-04 缺陷件 100% 被误判为正常。保留仅为记录，**不要当能力指标引用**。
CV_NORMAL_RECALL = 1.000
#: ⚠️ 6 类**整体**折外 macro-F1 未单独记录 → meta 里不写这个字段。
#:   按纪律「数字做完才填」，宁可缺字段也不填估算值。


class InferError(RuntimeError):
    pass


@dataclass
class Prediction:
    pred_class: str
    proba: float
    top2: list          # [(class, proba), ...]
    kb_ids: list
    kb_details: list
    rejected: bool
    reject_reason: str
    #: 源图片路径（pipeline 填，便于报告与回溯）
    source_path: str = ""
    #: 用户可见文案。**只在正常类命中时有值**（其余场景为空，
    #: 因为缺陷类的话由kb_details 里的根因承载，不在这里重复编）。
    advice: str = ""
    #: 是否为「未检出缺陷」这一确定结论。
    #: 🔴 与 `rejected` 是**不同语义**：正常 = 答出来了（不必拒答），
    #:   拒答 = 答不出来。下游若把两者混为一谈，正常件会被当成失败。
    is_normal: bool = False

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "pred_class": self.pred_class,
            "proba": round(self.proba, 4),
            "top2": [(c, round(p, 4)) for c, p in self.top2],
            "kb_ids": self.kb_ids,
            "kb_details": self.kb_details,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "advice": self.advice,
            "is_normal": self.is_normal,
        }


# ---------------------------------------------------------------- 特征提取
def build_backbone(device: str = "cuda"):
    import torch
    from src.vision.weights import build_model
    m = build_model(BACKBONE, num_classes=0, img_size=IMG).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _tf():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG, IMG)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def embed_paths(paths, model=None, device: str = "cuda", batch: int = 64) -> np.ndarray:
    """任意张图片 → [N, 1280] 冻结特征。**这是推理端唯一的图像入口**。"""
    import torch
    from PIL import Image
    own = model is None
    model = model or build_backbone(device)
    tf = _tf()
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            xs = []
            for p in paths[s:s + batch]:
                with Image.open(p) as im:
                    xs.append(tf(im.convert("RGB")))
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                f = model.forward_features(torch.stack(xs).to(device))
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            elif f.dim() == 3:
                f = f[:, 0]
            out.append(f.float().cpu())
    if own:
        del model
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 正常件来源
def _dc04_normal_names(z, n: int, seed: int = 42) -> list:
    """从打开的 DC-04 zip 里挑 n 个正常件**场景**，每场景返回 1 个视图的条目名。

    🔴 **按场景分组，绝不能随机取图**：DC-04 有 cropped / full_res 两套视图，
       **同名文件 = 同一场景**（实测 7,006 个文件名 100% 可配对）。
       同场景两视图同时进训练集 → 折外评估泄漏。
       D8 已踩过这个坑（独立抽样导致配对数悄悄归零，核心实验失效）。
    """
    import random
    by_scene: dict = {}
    for nm in z.namelist():
        if not nm.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = nm.split("/")
        # 结构：test/{视图}/no_error_print/camera{1,2,3}/detection_camN_XXXX.jpg
        if len(parts) < 5 or parts[2] != "no_error_print":
            continue
        scene = parts[-1].rsplit("_", 1)[-1].split(".")[0]
        by_scene.setdefault(scene, []).append(nm)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    picked = rng.sample(scenes, min(n, len(scenes)))
    return [rng.choice(by_scene[s]) for s in picked]


def _embed_from_zip(z, names: list, device: str, batch: int = 64) -> np.ndarray:
    """从已打开的 zip 里逐张解码 → 骨干提特征。

    🔴 与 `embed_paths` **共用同一套前向**（同一 `_tf()` + `build_backbone`），
       否则正常件与 DC-02R 的特征不在同一空间，LogReg 会学到「域差异」而非「缺陷形态」。
       本项目最贵的教训就是「口径不一致时模型会静默学错东西」。
    """
    import io
    import torch
    from PIL import Image
    model = build_backbone(device)
    tf = _tf()
    out = []
    with torch.no_grad():
        for s in range(0, len(names), batch):
            xs = []
            for nm in names[s:s + batch]:
                im = Image.open(io.BytesIO(z.read(nm))).convert("RGB")
                xs.append(tf(im))
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                f = model.forward_features(torch.stack(xs).to(device))
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            elif f.dim() == 3:
                f = f[:, 0]
            out.append(f.float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 训练与落盘
def train_and_save(folds_json: str = None, out_dir: Path = None,
                   device: str = "cuda", verbose: bool = True,
                   n_normal: int = N_NORMAL_TRAIN,
                   dc04_zip: str = None):
    """用**全量**数据重训 L2 并落盘（6 类：5 缺陷 + 正常件）。

    ⚠️ 落盘的是**部署件**，不是被测件。性能数字只能引用
       折外结果（CV_MACRO_F1_DEFECT_5），本函数不产出性能指标。

    🔴 2026-10-06 起混入 DC-04 正常件（默认 200 个场景）：
       此前是 5 类强制分类 → 好件必然被误报（域外实测 72%~75%）。
       `n_normal=0` 可退回旧行为（仅供对照，不要用于部署）。
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir or MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fj = Path(folds_json or (ROOT / "data" / "p1_folds.json"))
    d = json.loads(fj.read_text(encoding="utf-8"))
    items = d["images"]
    paths = [it["path"] for it in items]
    # 🔴 标签用 **DEFECT_CLASSES**（5 类）而不是 CLASSES（6 类）——
    #   DC-02R 里没有正常件，若拿 6 类列表去 index，必然 ValueError；
    #   更隐蔽的是若有人"顺手"把 No_error 塞进 DC-02R 的标签里，
    #   就会把域外图片混进主训练集。类别归属必须由数据源决定，不能由模型决定。
    y = np.array([DEFECT_CLASSES.index(it["label"]) for it in items])

    if verbose:
        print(f"[train] DC-02R 全量：{len(paths)} 张 / {len(DEFECT_CLASSES)} 类"
              f"（部署件，不产出性能指标）")
    F = embed_paths(paths, device=device)

    # ---- 混入 DC-04 正常件（第 6 类）----
    n_normal_used = 0
    zp_used = None
    if n_normal and n_normal > 0:
        import zipfile
        zp = Path(dc04_zip) if dc04_zip else DC04_ZIP
        zp_used = zp
        with zipfile.ZipFile(zp) as z:
            Fn = _embed_from_zip(z, _dc04_normal_names(z, n_normal), device)
        yn = np.full(len(Fn), len(DEFECT_CLASSES), dtype=int)   # 正常类索引 = 5
        F = np.concatenate([F, Fn], axis=0)
        y = np.concatenate([y, yn], axis=0)
        n_normal_used = len(Fn)
        if verbose:
            print(f"[train] + DC-04 正常件 {n_normal_used} 张→ 第 6 类 "
                  f"({NORMAL_CLASS})，合计 {len(F)} 张 / {len(CLASSES)} 类")

    sc = StandardScaler().fit(F)
    clf = LogisticRegression(max_iter=3000, C=1.0,
                             class_weight="balanced").fit(sc.transform(F), y)

    # 🔴 落盘前自检：分类头列数必须等于**实际训练用到的**类别数。
    #   历史上出现过「训练集缺类 → clf 只输出部分列 → 推理时索引错位且不报错」。
    #   🔴 注意基准是 `classes_eff` 而非 `CLASSES`：正常类默认关闭（n_normal=0），
    #   此时分类头本就该是 5 类。若拿 6 类当基准，正常类一关闭就误报——
    #   **护栏本身也会说谎**（本次实测踩到）。
    classes_eff = DEFECT_CLASSES + ([NORMAL_CLASS] if n_normal_used else [])
    if len(clf.classes_) != len(classes_eff):
        raise InferError(
            f"分类头输出 {len(clf.classes_)} 类，但本次训练实际用了 "
            f"{len(classes_eff)} 类：{list(clf.classes_)} vs {classes_eff}。\n"
            f"→ 若 n_normal>0，说明正常件没能进来（检查 DC-04 zip 可达性）；"
            f"若 n_normal=0，说明训练数据里混进了意料之外的类别。")

    guard = _sanity_check_ood(sc, clf, zp_used, device,
                             enabled=bool(n_normal_used), verbose=verbose)

    joblib.dump({"scaler": sc, "clf": clf, "classes": classes_eff},
                out_dir / "l2_deploy.joblib")

    # 骨干权重：复制而非重下（权重加载是本项目最大踩坑区，别在推理侧再触发）
    src_w = _locate_backbone_weight()
    if src_w:
        shutil.copy2(src_w, out_dir / "backbone.safetensors")

    meta = {
        "backbone": BACKBONE,
        "img_size": IMG,
        "n_train_dc02r": len(paths),
        "n_train_normal": n_normal_used,
        "n_train_total": len(F),
        "classes": classes_eff,
        "n_defect_classes": len(DEFECT_CLASSES),
        "normal_class_enabled": bool(n_normal_used),
        "n_features": int(F.shape[1]),
        "level": "L2 (frozen backbone + StandardScaler + LogisticRegression)",
        "hyperparams": {"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
        "normal_class_source": {
            "dataset": "DC-04 (Zenodo 18628777)",
            "class_in_dataset": "no_error_print",
            "license": "CC BY 4.0（**需署名**）",
            "n_used": n_normal_used,
            "🔴_status": (
                "**默认不启用（n_normal=0）**。落地实测发现：混入 DC-04 正常件后，"
                "**DC-04 的缺陷件也有 100% 被判成 No_error**（p≈0.999），"
                "因为 DC-04 三类质心与 DC-02R 全体质心余弦 0.53~0.55、几乎重合，"
                "正常件仅领先 +0.019。模型学到的是「像 DC-04 ⇒ 正常」而非"
                "「无缺陷 ⇒ 正常」。class_weight=None 对此无改善（非权重问题）。"
                "→ 正确解法是**自采与目标域同分布的正常件**。"),
            "sampling": "按**场景**抽样，每场景取 cropped/full_res 之一"
                       "（同名文件 = 同一场景，同进训练集会造成折外泄漏）",
            "why_not_in_training_defects": (
                "DC-04 的 layershift/extrusion 与 DC-02R 的 Layer_shifting/"
                "Off_platform 语义只是近似，实测特征可分性仅 70.8%。"
                "把它们当训练数据 = 把「近似当等同」固化进模型。"),
        },
        "expected_cv_macro_f1": CV_MACRO_F1,
        "expected_cv_macro_f1_defect_5": CV_MACRO_F1_DEFECT_5,
        "expected_cv_normal_recall": CV_NORMAL_RECALL,
        "expected_cv_source": (
            "exp/p1_d3_L2_efficientnet_b0.json（5 类 / group-aware 5 折 × 2 种子，折外）；"
            "6 类口径见 exp/d9_class_weight_check.json（**仅折外，不可搬进部署件**）"),
        "known_limitation": (
            "⚠️ **本模型没有域适应能力。** DC-04 实测（D8）：跨机型/相机/构图"
            "（Prusa i3 + 工业相机 + 红色立方件）时输出塌缩——86.6% 的图判成 "
            "Stringing，真值正常件也有 91.0% 判成 Stringing。"
            "🔴 **且正常类无法用 DC-04 正常件补**（见 normal_class_source）："
            "那样会让 DC-04 缺陷件 100% 被误判为正常——"
            "**比原塌缩更危险**（原错误还会给排查建议，全判正常则一条建议都不给）。"
            "→ 换机型/换材料必须**自采该域的正常件与缺陷件**并重新标定；"
            "→ 长期需引入 **OOD/域检测且必须先于分类器**。"),
        "eval_note": (
            f"⚠️ 本模型在**全量 {len(F)} 张**上重训"
            f"（DC-02R {len(paths)} + DC-04 正常件 {n_normal_used}），用于部署。"
            "**不要**用它算 macro-F1 作为性能指标——那是训练集准确率，必然虚高。"
            f"合法性能数字只有折外的：DC-02R 5 类 {CV_MACRO_F1_DEFECT_5:.2%}、"
            f"DC-04 正常件识别 {CV_NORMAL_RECALL:.1%}"
            "（来自 exp/d9_class_weight_check.json）。"),
        "folds_json_sha256": _sha256(fj),
        "ood_guard": guard,
        "kb_mapping": "src/vision/kb_mapping.py（改映射后无需重训，映射是纯查表）",
    }
    (out_dir / "l2_deploy.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[train] 已落盘 {out_dir}/"
              f"（{'含' if src_w else '无'}骨干权重）")
    return meta


def _sanity_check_ood(sc, clf, dc04_zip, device: str, n: int = 40,
                      seed: int = 7, enabled: bool = True,
                      verbose: bool = True) -> dict:
    """落盘前的**域外健全性护栏**：用 DC-04 缺陷件检查是否被误判成正常类。

    🔴🔴 **为什么必须有这道护栏（2026-10-06 实测踩出来的）**：
        D9 的 A/B 实验只测了「正常件能否被认出」，**没测「缺陷件会不会被
        一起认成正常」**。落地后端到端回归立刻暴露：

            DC-04 正常件→ No_error  p=0.9996  ✅
            DC-04 layershift → No_error  p=0.9994  🔴 全错
            DC-04 extrusion  → No_error  p=0.9973  🔴 全错

        即 **DC-04 上 100% 的图都被判成正常件**。
        原因（已用特征空间测量定位，不是猜）：
          · DC-04 三类质心与 DC-02R 全体质心的余弦 = 0.53~0.55，
            **三类彼此几乎重合**，域外特征挤在同一点；
          · 正常件质心相对缺陷类质心只领先 +0.019，
            而 `class_weight="balanced"` 又把它放大约 1.76倍；
          → 于是模型学到的其实是「**像 DC-04** ⇒ 正常」，
            而不是「**无缺陷** ⇒ 正常」。跨域时这条捷径必然压过真实信号。

        ⚠️ **这不推翻 D9 的结论**，而是划定了它的边界：
        D9 测的是「DC-02R 缺陷类 + DC-04 正常件」这个混合分布上的行为，
        **从没有验证「DC-04 缺陷件不会被误判为正常」**。
        少测的那一项恰好是上线后最致命的一项。

    故：**落盘前必须过这道检查**，否则拒绝写文件。
    这与「部署件不能报 99.73%」是同一类护栏——
    不是阻止 Reporting，而是防止**未验证的模型被当成已验证的交付出去**。

    判据（先写死，避免看到结果再挑说法）：
        DC-04 缺陷件被判成 No_error 的比例必须 < 50%。
        超过则说明正常类已吞掉缺陷域→ **拒绝落盘**。
    ⚠️ 阈值 50% 是「是否发生了类别吞噬」的性质判定，不是性能指标；
       真实能力上限仍由 D8 结论给出（域外缺陷件之间只分得开 70.8%）。
    """
    if not enabled:
        # 正常类没启用 → 不存在「被正常类吞噬」的风险，无需检查。
        # ⚠️ 但DC-04 缺陷件仍会被判成某个缺陷类（原D8 塌缩），那是**已知**局限，
        #    由 meta 的 known_limitation 记录，不在本护栏范围内。
        return {"skipped": "正常类未启用（n_normal=0），无吞噬风险"}
    if dc04_zip is None or not Path(dc04_zip).exists():
        return {"skipped": "DC-04 zip 不可达，跳过域外护栏"}
    import random
    import zipfile
    nrm = len(DEFECT_CLASSES)
    rng = random.Random(seed)
    names = []
    with zipfile.ZipFile(dc04_zip) as z:
        for cat in ("layershift_error_print", "extrusion_error_print"):
            pool = [x for x in z.namelist()
                    if x.lower().endswith((".jpg", ".jpeg", ".png"))
                    and f"/cropped/{cat}/" in x]
            names += rng.sample(pool, min(n // 2, len(pool)))
        Fd = _embed_from_zip(z, names, device)
    pred = clf.predict(sc.transform(Fd))
    as_normal = float((pred == nrm).mean())
    res = {"n_checked": len(names), "pred_as_normal_rate": as_normal,
           "threshold": 0.5, "passed": as_normal < 0.5}
    if verbose:
        print(f"[guard] 域外健全性：DC-04 缺陷件 {len(names)} 张 → "
              f"被误判成 {NORMAL_CLASS} 的比例 {as_normal*100:.1f}%"
              f"（阈值 50%）")
    if not res["passed"]:
        raise InferError(
            f"拒绝落盘：DC-04 缺陷件有 **{as_normal*100:.1f}%** 被判成 "
            f"`{NORMAL_CLASS}`（阈值 50%）。\n"
            f"→ 正常类已经**吞掉了缺陷域**：模型学到的是「像 DC-04 ⇒ 正常」，\n"
            f"  而不是「无缺陷 ⇒ 正常」。上线后表现为**所有图都报「未发现缺陷」**，\n"
            f"  比原来的「都报Stringing」更危险——原错误至少还会给排查建议。\n"
            f"→ 根因（已测特征空间）：DC-04 三类质心与 DC-02R 全体质心余弦 0.53~0.55，\n"
            f"  三类几乎重合；正常件仅领先 +0.019，balanced 又放大 1.76×。\n"
            f"→ 可选处置：① 换与目标域同分布的正常件（**推荐**）；\n"
            f"  ② class_weight 改默认（实测DC-02R 5 类 -0.13pp，仍在噪声内）；\n"
            f"  ③ 暂时 n_normal=0 退回 5 类。**别靠调阈值解决**——\n"
            f"  塌缩下高置信度是危险信号，阈值救不了类别吞噬。")
    return res


def _locate_backbone_weight():
    """从 HF cache 找 EfficientNet-B0 权重（只复制，不下载）"""
    for base in [Path.home() / ".cache/huggingface/hub",
                 Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None]:
        if not base or not base.exists():
            continue
        for p in base.glob("models--timm--efficientnet_b0*/blobs/*"):
            if p.is_file() and p.stat().st_size > 1_000_000:
                return p
    return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 加载与预测
def load_deployed_model(out_dir: Path = None, device: str = "cuda"):
    """加载部署件。**一致性校验失败直接抛错**，不静默降级。"""
    import joblib
    out_dir = Path(out_dir or MODEL_DIR)
    jb = out_dir / "l2_deploy.joblib"
    if not jb.exists():
        raise InferError(
            f"部署件不存在: {jb}\n先跑 `python -m src.vision.infer --train`")
    blob = joblib.load(jb)
    # 🔴 允许两种合法配置：5 类（默认，正常类关闭）与 6 类（正常类启用）。
    #   顺序必须严格等于前缀关系——列索引与类别名一一对应，错位会静默取错列。
    got = list(blob["classes"])
    if got not in (list(DEFECT_CLASSES), list(CLASSES)):
        raise InferError(
            f"部署件类别与本模块不一致:\n  落盘={got}\n"
            f"  允许={DEFECT_CLASSES}（5 类）\n     或={CLASSES}（6 类）\n"
            f"→ 多数是 `CLASSES` 顺序被改动，会让 proba 列索引与类别名**静默错位**。"
            f"请重跑 `python -m src.vision.infer --train`。")
    meta_p = out_dir / "l2_deploy.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        cur = _sha256(ROOT / "data" / "p1_folds.json")
        if meta.get("folds_json_sha256") and meta["folds_json_sha256"] != cur:
            print(f"[infer] ⚠️ 警告：训练用的 folds.json 已变更"
                  f"（{meta['folds_json_sha256']} → {cur}），"
                  f"部署件可能与当前折定义不一致")
    else:
        meta = {}
    # 🔴 实际类别必须透出给pipeline：5 类部署件下 proba 只有 5 列，
    #   若 pipeline 拿模块级 CLASSES(6) 去索引，`CLASSES[top1]` 在 top1=5 时
    #   要么 IndexError、要么（更糟）静默取到错误类别名。
    meta["classes"] = got
    meta["normal_class_enabled"] = NORMAL_CLASS in got
    model = build_backbone(device)
    return blob["scaler"], blob["clf"], model, meta


def predict_probs(paths, scaler, clf, model=None, device: str = "cuda") -> np.ndarray:
    F = embed_paths(paths, model=model, device=device)
    return clf.predict_proba(scaler.transform(F))


# ---------------------------------------------------------------- 评估护栏
def evaluate(model_dir: Path = None, refuse_train_data: bool = True):
    """⚠️ 护栏函数：拒绝在全量部署件上算「性能」。

    有人调用 `evaluate()` 大概率是想拿一个数字汇报，
    而在全量模型上算出来的必然是训练集准确率 → 会污染报告口径。
    故默认直接抛错，要算必须在**折外**做（用 train.py）。
    """
    meta_p = Path(model_dir or MODEL_DIR) / "l2_deploy.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if refuse_train_data:
            raise InferError(
                "拒绝执行：`data/_model/` 里是**部署件**（全量重训），"
                "在它上面算的 macro-F1 是训练集准确率，不能作为性能指标。\n"
                f"合法口径：折外 {meta['expected_cv_macro_f1']:.2%}"
                f"（{meta['expected_cv_source']}）\n"
                "若要新的折外数字，请跑 `python -m src.vision.train "
                "--level L2 --backbone efficientnet_b0 --scheme grouped --seeds 42,43`")


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="全量重训并落盘")
    ap.add_argument("--predict", nargs="+", help="对若干图片预测并输出 KB 条目")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-normal", type=int, default=N_NORMAL_TRAIN,
                    help=f"混入多少个 DC-04 正常件场景（第 6 类），默认 {N_NORMAL_TRAIN}。"
                         "0 =退回 5 类强制分类（仅供对照，**不要用于部署**）")
    ap.add_argument("--dc04-zip", default=None,
                    help=f"DC-04 zip 路径，默认 {DC04_ZIP}")
    ap.add_argument("--reject-threshold", type=float, default=0.90,
                    help="低于此概率则拒答。默认 0.90 = 已用**折外**预测标定的值"
                         "（pipeline.DEFAULT_REJECT），不是拍脑袋")
    a = ap.parse_args()

    if a.train:
        train_and_save(out_dir=a.out_dir, device=a.device,
                       n_normal=a.n_normal, dc04_zip=a.dc04_zip)
        return 0

    if a.predict:
        from src.vision.pipeline import diagnose
        res = diagnose(a.predict, device=a.device,
                       reject_threshold=a.reject_threshold)
        print(json.dumps([r.to_dict() for r in res],
                         ensure_ascii=False, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
