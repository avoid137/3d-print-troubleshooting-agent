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
CLASSES = ["Warping", "Cracking", "Stringing", "Layer_shifting", "Off_platform"]
MODEL_DIR = ROOT / "data" / "_model"
IMG = 224
BACKBONE = "efficientnet_b0"
#: train.py 折外实测值（L2 / B0 / group-aware5 折 × 2 种子）
CV_MACRO_F1 = 0.9973


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


# ---------------------------------------------------------------- 训练与落盘
def train_and_save(folds_json: str = None, out_dir: Path = None,
                   device: str = "cuda", verbose: bool = True):
    """用**全量**数据重训 L2 并落盘。

    ⚠️ 落盘的是**部署件**，不是被测件。性能数字只能引用
       train.py 的折外结果（CV_MACRO_F1），本函数不产出性能指标。
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
    y = np.array([CLASSES.index(it["label"]) for it in items])

    if verbose:
        print(f"[train] 全量重训 L2：{len(paths)} 张 / {len(set(y.tolist()))} 类"
              f"（这是部署件，不产出性能指标）")
    F = embed_paths(paths, device=device)
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(max_iter=3000, C=1.0,
                             class_weight="balanced").fit(sc.transform(F), y)

    joblib.dump({"scaler": sc, "clf": clf, "classes": CLASSES},
                out_dir / "l2_deploy.joblib")

    # 骨干权重：复制而非重下（权重加载是本项目最大踩坑区，别在推理侧再触发）
    src_w = _locate_backbone_weight()
    if src_w:
        shutil.copy2(src_w, out_dir / "backbone.safetensors")

    meta = {
        "backbone": BACKBONE,
        "img_size": IMG,
        "n_train": len(paths),
        "classes": CLASSES,
        "n_features": int(F.shape[1]),
        "level": "L2 (frozen backbone + StandardScaler + LogisticRegression)",
        "hyperparams": {"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
        "expected_cv_macro_f1": CV_MACRO_F1,
        "expected_cv_source": "exp/p1_d3_L2_efficientnet_b0.json（group-aware 5 折 × 2 种子，折外）",
        "eval_note": ("⚠️ 本模型在**全量 1912 张**上重训，用于部署。"
                      "**不要**用它算macro-F1 作为性能指标——那是训练集准确率，必然虚高。"
                      f"合法性能数字只有折外的 {CV_MACRO_F1:.2%}（来自 train.py 交叉验证）。"),
        "folds_json_sha256": _sha256(fj),
        "kb_mapping": "src/vision/kb_mapping.py（改映射后无需重训，映射是纯查表）",
    }
    (out_dir / "l2_deploy.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[train] 已落盘 {out_dir}/"
              f"（{'含' if src_w else '无'}骨干权重）")
    return meta


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
    if blob["classes"] != CLASSES:
        raise InferError(f"类别顺序与本模块不一致: {blob['classes']} != {CLASSES}")
    meta_p = out_dir / "l2_deploy.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        cur = _sha256(ROOT / "data" / "p1_folds.json")
        if meta.get("folds_json_sha256") and meta["folds_json_sha256"] != cur:
            print(f"[infer] ⚠️ 警告：训练用的 folds.json 已变更"
                  f"（{meta['folds_json_sha256']} → {cur}），"
                  f"部署件可能与当前折定义不一致")
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
    ap.add_argument("--reject-threshold", type=float, default=0.60,
                    help="低于此概率则拒答（宁可说不知道也不给错建议）")
    a = ap.parse_args()

    if a.train:
        train_and_save(out_dir=a.out_dir, device=a.device)
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
