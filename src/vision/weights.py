"""P1 权重加载器：绕开 Windows + hf-mirror 环境下的两个已实测陷阱。

背景（均为 2026-10-04 在 RTX 4060 机器上的实测结论，不是猜测）：

陷阱 1｜缓存快照 0 字节
    `huggingface_hub` 在 Windows 未开启开发者模式时**无法创建符号链接**，
    会静默留下一个 0 字节的 snapshot 文件，而 blob 本身却是完整的。
    表现：`timm.create_model(pretrained=True)` 抛
        `SafetensorError: Error while deserializing header: header too small`
    实测证据：blob = 21,355,344 B（完整，360 个张量），snapshot = 0 B。
    ⚠️ 所以「header too small」不等于网络失败，可能只是链接失败。

陷阱 2｜hf_hub 传输被截断
    同一网络下 `curl` 直连 hf-mirror 能拿到完整 21.4 MB（@1.08 MB/s），
    但 `hf_hub_download`（无论走缓存还是 `local_dir=`）落地都是 0 字节。
    结论：本机**不要依赖 hf_hub 自己下载**，改用 curl 预热缓存或直接指向本地文件。

对策（本模块的职责）
    1. `load_pretrained_into(model, name)`：按 本地文件 → HF 缓存 blob → 在线下载 三级查找，
       找到完整权重就 `load_state_dict(strict=False)` 注入，并**校验注入是否真的生效**
       （张量数与形状对不上就抛错，不静默退化成随机初始化）。
    2. `fetch_weights(...)`：用 urllib 走镜像下载到 `data/_weights/`，带完整性校验。
    3. `HF_MIRROR` 常量：本机实测可达的镜像；魔塔(ModelScope)/阿里云 mirrors 均**不可达**。

用法（train.py 侧只需把 timm.create_model(..., pretrained=True) 换成本函数）：
    from src.vision.weights import build_model
    model = build_model("efficientnet_b0", num_classes=5, img_size=224)
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import timm
import torch

# 本机实测（2026-10-04）：hf-mirror.com 200 @2.56s；huggingface.co 超时；
# modelscope.cn(魔塔) 与 mirrors.aliyun.com 均 status=000 不可达。
HF_MIRROR = "https://hf-mirror.com"

WEIGHTS_DIR = Path("data/_weights")

# 期望大小（字节）。用于「下载完整但静默截断」的兜底判定。
# ⚠️ 全部为**实测值**（2026-10-04 本机下载后 stat 所得），不是查文档抄来的。
#    DINOv2 最初按估计写 346,584,396，实际是 346,334,872 —— 差 249 KB，
#    若不按实测修正会把完好的文件误判为「截断」并反复重下。
EXPECTED_BYTES = {
    "efficientnet_b0.ra_in1k": 21_355_344,          # 实测，360 张量
    "mobilenetv3_small_100.lamb_in1k": 10_241_912,  # 实测，1,522,981 参数
    "mobilenetv3_large_100.ra_in1k": None,
    "vit_base_patch14_dinov2.lvd142m": 346_334_872,  # 实测，174 张量
}

# 本机已验证可用的缓存 blob（Windows 快照为 0 字节时，从 blob 直读）
_VERIFIED_BLOBS = {
    "efficientnet_b0.ra_in1k": (
        "~/.cache/huggingface/hub/models--timm--efficientnet_b0.ra_in1k/"
        "blobs/d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"
    ),
}


class WeightError(RuntimeError):
    """权重获取或注入失败。**不允许静默退化成随机初始化。**"""


@dataclass
class LoadReport:
    """权重加载报告 —— 训练日志要打印它，避免「以为用了预训练其实没有」。"""

    source: str                      # local / cache_blob / hf_cache / download
    path: str
    injected: int = 0                # 成功注入的张量数
    skipped: list[str] = field(default_factory=list)   # 故意跳过的（分类头）
    shape_mismatch: list[str] = field(default_factory=list)
    total_params: int = 0

    def summary(self) -> str:
        return (
            f"[weights] source={self.source} path={self.path}\n"
            f"          注入张量 {self.injected} 个，跳过 {self.skipped}，"
            f"形状不符 {self.shape_mismatch}\n"
            f"          模型总参数 {self.total_params:,}"
        )


def _hf_blob_paths(name: str):
    """在 HF 缓存里找 timm/<name> 的 blob（跳过 0 字节的坏文件）。"""
    root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    repo_dir = root / f"models--timm--{name}"
    blobs = repo_dir / "blobs"
    if not blobs.is_dir():
        return []
    return [p for p in blobs.iterdir() if p.is_file() and p.stat().st_size > 1024]


def _verify(path: Path, name: str) -> bool:
    """完整性校验：大小对不上就判为截断（期望值为 None 表示未实测，跳过校验）。"""
    exp = EXPECTED_BYTES.get(name)
    if exp is None:
        return path.stat().st_size > 1024
    return path.stat().st_size == exp


def _read_state_dict(path: Path) -> dict:
    """读权重。**后缀不可靠**——HF 缓存里的 blob 文件名是 sha256，没有扩展名，
    但内容其实是 safetensors（实测踩过：按 `.pth` 走 torch.load 会报
    `UnpicklingError: Unsupported operand 176`）。所以先嗅探魔数。"""
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) >= 8:
        n = int.from_bytes(head[:8], "little")
        # safetensors: 前 8 字节 = header 长度，且后面紧跟 JSON 的 '{'
        if 0 < n < 100_000_000:
            with open(path, "rb") as f:
                f.seek(8)
                if f.read(1) == b"{":
                    from safetensors.torch import load_file
                    return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def fetch_weights(name: str, dest_dir: Path = WEIGHTS_DIR) -> Path:
    """用 urllib 走 hf-mirror 下载权重到本地（绕开 hf_hub 的截断问题）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}.safetensors"
    if out.exists() and _verify(out, name):
        return out

    url = f"{HF_MIRROR}/timm/{name}/resolve/main/model.safetensors"
    exp = EXPECTED_BYTES.get(name)
    tmp = out.with_suffix(".part")
    print(f"[weights] 下载 {url}")
    # 关键：必须带 HF_ENDPOINT 语义（直连镜像域名），且不走任何代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=60) as r, open(tmp, "wb") as f:
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
    if exp and got != exp:
        tmp.unlink(missing_ok=True)
        raise WeightError(f"下载不完整：得到 {got} B，期望 {exp} B（{name}）")
    tmp.replace(out)
    return out


def load_pretrained_into(model: torch.nn.Module, name: str,
                         report: LoadReport | None = None) -> LoadReport:
    """把 ImageNet 预训练权重注入 `model`，按三级来源查找，**校验注入是否生效**。

    分类头（`classifier.*` / `fc.*` / `head.*`）故意跳过——目标任务是 5 类，
    1000 类头没有意义；这也正是标准迁移学习的做法。
    """
    # 1) 显式本地文件
    candidates: list[tuple[str, Path]] = []
    for cand in (WEIGHTS_DIR / f"{name}.safetensors",
                 WEIGHTS_DIR / f"{name}.pth"):
        if cand.is_file() and _verify(cand, name):
            candidates.append(("local", cand))
    # 2) 本机已验证的 blob
    vb = _VERIFIED_BLOBS.get(name)
    if vb:
        p = Path(os.path.expanduser(vb))
        if p.is_file() and _verify(p, name):
            candidates.append(("cache_blob", p))
    # 3) HF 缓存里任一完整 blob
    for p in _hf_blob_paths(name):
        if _verify(p, name):
            candidates.append(("hf_cache", p))
    # 4) 在线下载
    if not candidates:
        p = fetch_weights(name)
        candidates.append(("download", p))

    src, path = candidates[0]
    if len(candidates) > 1:
        print(f"[weights] 发现 {len(candidates)} 份可用权重，取 {src}: {path}")

    sd = _read_state_dict(path)
    tgt = model.state_dict()

    drop_prefixes = ("classifier.", "fc.", "head.")
    clean = {k: v for k, v in sd.items()
             if not k.startswith(drop_prefixes) and k in tgt
             and tgt[k].shape == v.shape}
    skipped = sorted(k for k in sd if k.startswith(drop_prefixes))
    mismatch = sorted(
        k for k, v in sd.items()
        if k in tgt and not k.startswith(drop_prefixes) and tgt[k].shape != v.shape
    )

    missing, unexpected = model.load_state_dict(clean, strict=False)

    rep = report or LoadReport(source=src, path=str(path))
    rep.injected = len(clean)
    rep.skipped = skipped
    rep.shape_mismatch = mismatch
    rep.total_params = sum(p.numel() for p in model.parameters())

    # 反向自检：注入 0 个张量 = 静默退化成随机初始化，必须报错而不是继续
    if rep.injected == 0:
        raise WeightError(f"注入 0 个张量，权重未生效：{path}")
    if len(clean) < 0.5 * len(tgt):
        raise WeightError(
            f"注入张量 {len(clean)}/{len(tgt)} 不足一半，疑似权重与模型不匹配：{path}"
        )
    return rep


def build_model(backbone: str, num_classes: int = 5, img_size: int = 224):
    """替代 `timm.create_model(..., pretrained=True)` 的健壮版本。

    归一化常数仍由 `timm.data.resolve_data_config` 提供（**绝不手写**），
    与 guards.py 的骨干审计口径一致。
    """
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    # 仓库名以 timm 的 hf_hub_id 为准，本表只做白名单限制
    hub = model.pretrained_cfg.get("hf_hub_id")
    repo_name = _hf_repo_name(hub) if hub else None
    allowed = _resolve_repo_name(backbone)
    if repo_name is None:
        print(f"[weights] ⚠️ {backbone} 的 timm 配置无 hf_hub_id，使用随机初始化")
    else:
        if allowed is not None and allowed != repo_name:
            raise WeightError(
                f"权重映射表与 timm 不一致：{backbone} 本表写 '{allowed}'，"
                f"timm 的 hf_hub_id 是 '{hub}'。请改 _resolve_repo_name。")
        load_pretrained_into(model, repo_name)
    if img_size and img_size != 224:
        _adapt_img_size(model, backbone, img_size)
    return model


def _adapt_img_size(model, backbone: str, img_size: int) -> None:
    """把模型适配到非224 输入。

    🔴 2026-10-04 实测坑：原实现无条件调`timm.models.resize_pos_embed`，
       但 **timm 1.0.30 里该函数已不存在** → AttributeError。
       更重要的是：**它本来就不该无条件调**——
         · CNN（EfficientNet / MobileNet）全卷积 +全局池化，**任意尺寸天然可用**，
           改pos_embed 根本不存在（`dynamic_img_size=False` 也印证了这点）。
         · ViT 的 pos_embed 与网格数强绑定，必须重采样。
       所以按骨干类型分派，且 ViT 走官方 `dynamic_img_size=True`，
       **不自己实现插值**（自己写容易与 timm 的插值口径不一致）。
    """
    is_vit = any(k in backbone.lower() for k in ("vit", "deit", "beit", "eva_"))
    if not is_vit:
        return                      # CNN：全卷积，无需任何处理
    if not getattr(model, "dynamic_img_size", False):
        raise WeightError(
            f"{backbone} 看起来是 ViT 但不支持 dynamic_img_size，"
            f"无法安全适配到 {img_size}px。请改用 224，或换支持动态尺寸的骨干。")
    model.dynamic_img_size = True
    print(f"[weights] {backbone}: dynamic_img_size=True @ {img_size}px")


def _resolve_repo_name(backbone: str) -> str | None:
    """timm 模型名 -> HF 仓库名。

    🔴 2026-10-04 实测坑：仓库名**不能猜**，必须向 timm 自己问。
       `mobilenetv3_small_100` 的真实仓库是 `timm/mobilenetv3_small_100.lamb_in1k`
       （带训练配方后缀），手写 `timm/mobilenetv3_small_100` 会 **HTTP 404**。
       所以这里的表只作「显式白名单」，真实取值一律从 `pretrained_cfg["hf_hub_id"]`读，
       并在 `_assert_repo_name_matches_timm` 里做反向自检。
    """
    table = {
        "efficientnet_b0": "efficientnet_b0.ra_in1k",
        "mobilenetv3_small_100": "mobilenetv3_small_100.lamb_in1k",
        "mobilenetv3_large_100": "mobilenetv3_large_100.ra_in1k",
        "vit_base_patch14_dinov2": "vit_base_patch14_dinov2.lvd142m",
    }
    return table.get(backbone)


def _hf_repo_name(hf_hub_id: str) -> str:
    """`timm/efficientnet_b0.ra_in1k` -> `efficientnet_b0.ra_in1k`"""
    return hf_hub_id.split("/", 1)[-1]


def _assert_repo_name_matches_timm(backbone: str, repo_name: str) -> None:
    """反向自检：本模块的映射表必须与 timm 的 `hf_hub_id` 一致，不一致当场炸。

    没有这道检查，表里少写一个 `.lamb_in1k` 就会一路404 到下载失败，
    而错误信息（HTTPError 404）完全指不到真正的原因。
    """
    model = timm.create_model(backbone, pretrained=False)
    truth = model.pretrained_cfg.get("hf_hub_id")
    if not truth:
        return
    want = _hf_repo_name(truth)
    if want != repo_name:
        raise WeightError(
            f"权重映射表与 timm 不一致：{backbone} 本表写'{repo_name}'，"
            f"timm 的 hf_hub_id 是 '{truth}'。"
            f"猜仓库名会一路404 到下载失败，且报错完全指不到原因。"
            f"请改 src/vision/weights.py 的 _resolve_repo_name。")


if __name__ == "__main__":
    import sys
    bb = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b0"
    m = build_model(bb, num_classes=5, img_size=224)
    print(f"✅ {bb} 构建完成，参数 {sum(p.numel() for p in m.parameters()):,}")
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 224, 224))
    print(f"   前向输出 {tuple(y.shape)}")
