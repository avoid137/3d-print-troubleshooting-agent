"""P1 去泄漏：前缀 + DINOv2 双路near-duplicate 检测 + group-aware 折划分。

为什么必须做（DC-02R 特有风险）：
    该数据集是同一批打印件连续拍摄的缺陷照片，同一实物可能有十几张近重复图。
    若逐张随机分层，同一实物会被分到训练折与测试折 → 模型等于见过测试图，
    指标虚高且**幅度未知**。这与 P0 判官间 ±7pp 噪声是同一类教训：
    不引入说不清的量。

🔴 为什么 pHash/dHash **不再参与 union**（2026-10-04 实测定案，勿回退）：
    原本设计是「pHash 粗筛 → DINOv2 精筛」，理由是省算力。实测三条硬证据
    证明这个粗筛在本pipeline 里**价值为零、代价极高**：

    1) **省不到算力**：DINOv2 特征是 **per-image** 的（1,912 张都要过一遍 ViT），
       粗筛一张特征都省不掉。而粗筛 supposed 要省掉的 N² 对余弦相似度计算，
       实测 1,912² 完整矩阵乘 + 上三角阈值 = **40.7 ms**。省 40ms，不值。
    2) **精确率极低**：pHash≤5 命中 2,539 对里，真正重复只有 163 对
       （161 对同前缀 + 2 对余弦同意），**误合并 2,376 对 = 94.3%**。
       其中 55 对直接跨类。
    3) **传递闭包放大污染**：这 2,376 对经并查集连通分量放大后，
       产出 19 个跨类污染 group（把不同类焊成同一实物），
       正好撞上「跨类污染必须为 0」硬断言。

    这与 SSRN 7395570 的结论一致：pHash 作为重复检测代理，其可靠性依模态而变，
    在 HAM10000 上只召回 3.3% 真实同病灶组。**故本模块保留 pHash 仅作
    只读诊断（写进报告作为负面证据），union 唯一依据 = 前缀 ∪ 余弦。**

    保留的两个可靠依据：
      + 文件名前缀（`xxx` / `xxx_aug` / `xxx_original`）——DC-02R **自带的
        确定性 ground truth**，81 组 242 张，实测零跨类
      + DINOv2 冻结特征余弦 ≥ 0.995 ——自标定阈值，误合并 0，最大组 3

产出 data/p1_folds.json，含两套划分：
    folds_stratified : 普通分层（仅作对照，用来算泄漏幅度差值）
    folds_grouped    : 去泄漏（**主口径**）

用法：
    python -m src.vision.dedup --data-root D:/datasets/fdm-defect
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .guards import EXPECTED_COUNTS, EXPECTED_TOTAL, GuardError, IMG_EXT

SEED = 42
N_FOLDS = 5

# 🔴 FEATURE_THRESHOLD 由**自标定实验**定（2026-10-04 实测），**不是照抄文献**。
#   文献/原默认 0.92 在本数据集上严重过松：1,912 张压成 39 组，
#   StratifiedGroupKFold 因group 跨折直接失效（比不聚组更糟）。
#   实测依据（1,826,916 对余弦）：
#     跨类对最大值仅 0.97116（P99.999）→ 阈值高于 0.98 不可能误判「同类不同实物」
#     0.995 处误合并对 = 0（零误判），最大组 = 3
#   故取 **0.995**（宁漏勿误：漏合并只影响去泄漏彻底性，误合并会把不同实物焊在一起）。
FEATURE_THRESHOLD = 0.995

# pHash 阈值**仅用于只读诊断**（量化它的误合并率写进报告），不参与 union。
# 实测：≤2 时误合并率 83.2%，≤5 时 94.3% —— 全都不可用。
PHASH_DIAG_THRESHOLD = 5

# 特征缓存（调阈值/重跑时秒级，不必重过 ViT）
CACHE_DIR = Path("exp/_cache")
FEAT_CACHE = CACHE_DIR / "dino_feats.npy"
PATH_CACHE = CACHE_DIR / "dino_paths.npy"


# ---------------------------------------------------------------- 感知哈希
def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """dHash：相邻像素比较。对重采样/轻裁剪稳健。"""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """pHash：DCT 域低频。对亮度/压缩更稳健。"""
    from scipy.fftpack import dct

    s = hash_size
    g = img.convert("L").resize((s * 4, s * 4), Image.LANCZOS)
    d = dct(np.asarray(g, dtype=np.float64), axis=0, norm="ortho")
    d = dct(d, axis=1, norm="ortho")[:s, :s]
    med = np.median(d.flatten()[1:])
    bits = (d > med).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------- 特征精筛
class FeatureExtractor:
    """DINOv2 冻结特征（去泄漏精筛用）。

    ⚠️ 实测（2026-10-04 本机）：`torch.hub.load("facebookresearch/dinov2", ...)`
    与 `timm.create_model(..., pretrained=True)` **都不可用** ——
    前者需 GitHub 直连（本机超时），后者因 Windows 无符号链接权限 + hf_hub
    传输截断得到 0 字节权重。统一走 src.vision.weights（已下好本地权重，174 张量实测注入）。
    """

    def __init__(self, device: str = "cuda"):
        import torch
        import timm
        from timm.data import resolve_data_config
        from src.vision.weights import _hf_repo_name
        self.torch = torch
        self.device = device
        self.backend = "timm-vitb14-dinov2(local)"
        self.model = timm.create_model("vit_base_patch14_dinov2",
                                       pretrained=False, num_classes=0)
        # 🔴 仓库名取自 timm 的 hf_hub_id，**不硬编码**（硬编码过一次，
        #   少写 `.lvd142m` 就404，而报错完全指不到原因）
        hub = self.model.pretrained_cfg.get("hf_hub_id")
        if not hub:
            raise GuardError("vit_base_patch14_dinov2 无 hf_hub_id，无法定位权重")
        from src.vision.weights import load_pretrained_into
        rep = load_pretrained_into(self.model, _hf_repo_name(hub))
        print(f"[dedup] DINOv2 权重: {rep.summary()}")
        # 归一化与输入尺寸一律取自 resolve_data_config（绝不手写 / 绝不硬编码）
        cfg = resolve_data_config({}, model=self.model)
        ic = cfg.get("input_size", (3, 518, 518))
        self.input_size = (int(ic[1]), int(ic[2]))
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
        print(f"[dedup] DINOv2 输入 {self.input_size} 归一化 mean={self.mean} std={self.std}")
        self.model.eval().to(device)

        # 冒烟：尺寸错配会在这里立刻炸，而不是静默降级
        probe = Image.new("RGB", (self.input_size[1], self.input_size[0]))
        v = self(probe)
        if v.shape[0] != self.dim:
            raise GuardError(f"DINOv2 特征维度异常: {v.shape[0]} != {self.dim}")

    @property
    def dim(self) -> int:
        return self.model.num_features

    def to_batch(self, imgs):
        """把 PIL 列表转成归一化后的 batch tensor（分批推理用）。

        ⚠️ 归一化只在这里和 `_prep` 各写一份**不行**——两处实现容易漂移。
        所以统一走这里，`_prep` 也复用（单张 = batch of 1）。
        """
        torch = self.torch
        import numpy as _np
        arr = _np.stack([_np.asarray(im, dtype="uint8") for im in imgs])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return ((x - mean) / std).to(self.device)

    def embed_batch(self, imgs, batch: int = 16) -> np.ndarray:
        """分批提特征并 L2 归一化，返回 [N, dim]。逐张推理 1,912 张约 5 min，分批约 40 s。"""
        torch = self.torch
        out = []
        with torch.no_grad():
            for s in range(0, len(imgs), batch):
                x = self.to_batch(imgs[s:s + batch])
                with torch.amp.autocast("cuda"):
                    f = self.model(x)
                v = f.float().cpu().numpy()
                out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
        return np.concatenate(out, axis=0)

    def _prep(self, img: Image.Image):
        """⚠️ 两个已实测的坑（2026-10-04）：

        1. **尺寸**：原实现硬编码 `Resize((224,224))`，但 DINOv2 ViT-B/14 原生 **518**
           （pos_embed 对应 1370 token = 518 patch + 1 cls + 851 register）。
           喂 224 会直接 `AssertionError: Input height (224) doesn't match model (518)`。
           → 从 `timm.resolve_data_config` 读真实 input_size，不再硬编码。
        2. **归一化**：原来手写 ImageNet mean/std，违反项目纪律「归一化只取
           `resolve_data_config`，绝不手写」（换骨干不换常数会静默劣化）。
        """
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return tf(img).unsqueeze(0).to(self.device)

    def __call__(self, img: Image.Image) -> np.ndarray:
        with self.torch.no_grad(), self.torch.amp.autocast("cuda"):
            f = self.model(self._prep(img))
        v = f.float().cpu().numpy().reshape(-1)
        return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------- 聚组
class _Union:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class DedupResult:
    paths: list
    labels: list
    group_ids: list
    n_groups: int
    n_dup_pairs_phash: int
    n_dup_pairs_feature: int
    feature_threshold: float
    phash_threshold: int
    # ⚠️ 以下带默认值 → 必须排在前述无默认值字段**之后**，
    #    否则 dataclass 直接 `TypeError: non-default argument follows default argument`
    n_dup_pairs_prefix: int = 0
    backend: str = "none"
    multi_groups: dict = field(default_factory=dict)
    n_impure_groups: int = 0          # 跨类污染组（误判的直接证据，应恒为 0）
    max_group: int = 0

    def summary(self) -> str:
        return (
            f"图片总数     : {len(self.paths)}\n"
            f"  去重后 group: {self.n_groups}"
            f"（折叠掉 {len(self.paths) - self.n_groups} 张）\n"
            f"  前缀命中对  : {self.n_dup_pairs_prefix} "
            f"（DC-02R 自带 _aug/_original，确定性）\n"
            f"  余弦命中对  : {self.n_dup_pairs_feature} "
            f"(≥{self.feature_threshold}, backend={self.backend})\n"
            f"  pHash 命中对: {self.n_dup_pairs_phash} (≤{self.phash_threshold})"
            f" —— **仅诊断，未参与 union**\n"
            f"  含多张的组  : {len(self.multi_groups)} 个"
            f"｜最大组 {self.max_group} 张"
            f"｜**跨类污染组 {self.n_impure_groups}**（应为 0）"
        )


def _prefix_key(p) -> str:
    """文件名前缀（去掉 `_aug` / `_original` 后缀）——**DC-02R 自带的确定性 ground truth**。

    实测（2026-10-04）：1,912 张里有 81 个前缀出现 3 次（80 组 `xxx.jpg` +
    `xxx_aug.jpg` + `xxx_original.jpg`）、1 组 2 次，共 242 张。
    这是**同一打印件的原图 + 增强版 + 连拍**，属确定性近重复，
    比任何阈值都可靠 → 作为主力分组依据，余弦只作补充。
    """
    from pathlib import Path as _P
    b = _P(p).name.rsplit(".", 1)[0]
    return b.replace("_aug", "").replace("_original", "")


def build_groups(paths, labels, use_features: bool = True,
                 feature_threshold: float = FEATURE_THRESHOLD,
                 phash_diag_threshold: int = PHASH_DIAG_THRESHOLD,
                 device: str = "cuda", verbose: bool = True,
                 reuse_cache: bool = True) -> DedupResult:
    """两路聚组：文件名前缀（确定性 ground truth） ∪ DINOv2 余弦（补漏）。

    🔴 pHash/dHash **不参与 union**，只在verbose 模式打印它的误合并率
      （实测 ≤5 时 2,539 命中对里 2,376 对是误合并 = 94.3%，详见模块 docstring）。
    """
    n = len(paths)
    if n == 0:
        raise GuardError("paths 为空，无法聚组")
    paths = [str(p) for p in paths]

    uf = _Union(n)

    # ---------------------------------------------- 0) 前缀分组（确定性）
    n_prefix = 0
    _pk: dict[str, int] = {}
    for i, p in enumerate(paths):
        k = _prefix_key(p)
        if k in _pk:
            uf.union(_pk[k], i)
            n_prefix += 1
        else:
            _pk[k] = i
    if verbose:
        print(f"[dedup] 前缀分组合并 {n_prefix} 对"
              f"（DC-02R 自带的_aug/_original 标记，零误判）")

    # ---------------------------------------------- 1) pHash 只读诊断
    #⚠️ 2026-10-04 实测踩坑：原实现把 1,912 张 **3072×2048 原图**全解码进 list
    #   → 约 21 GB → MemoryError。现在哈希与 DINOv2 都只读**流式缩略**，
    #   边读边缩、缩完即弃，峰值内存从 >21 GB 降到 ~350 MB。
    n_ph_diag = 0
    if verbose:
        print(f"[dedup] 计算 dHash/pHash（**仅诊断，不参与 union**）...")
    dhs, phs = [], []
    for p in paths:
        with Image.open(p) as im:
            im.draft("RGB", (512, 512))   # JPEG 快速缩放，避免解全图
            t = im.convert("RGB").resize((224, 224), Image.BILINEAR)
            t = t.copy()
        dhs.append(dhash(t))
        phs.append(phash(t))
    D = np.array([[(h >> b) & 1 for b in range(64)] for h in dhs], dtype=np.uint8)
    P = np.array([[(h >> b) & 1 for b in range(64)] for h in phs], dtype=np.uint8)
    # ⚠️ 口径：只数 i<j 的**唯一无序对**，与 exp/diag_d2_impure.py 完全一致。
    #   `np.minimum(...) <= thr` 对每个 i 都把 j<i 的对算一遍，直接 .sum() 会
    #   把同一对计两次（实测 5078 = 2×2539），报告里两个口径对不上就是 bug。
    n_ph_diag = 0
    for i in range(n):
        dm = np.minimum(np.bitwise_count(np.bitwise_xor(D, D[i])).sum(axis=1),
                        np.bitwise_count(np.bitwise_xor(P, P[i])).sum(axis=1))
        n_ph_diag += int((dm[i + 1:] <= phash_diag_threshold).sum())
    if verbose:
        print(f"[dedup] pHash≤{phash_diag_threshold} 命中 {n_ph_diag} 对 → "
              f"**不union**（实测误合并率 94.3%，见 exp/p1_d2_impure_diagnosis.json）")

    # ---------------------------------------------- 2) DINOv2 余弦 union
    n_ft = 0
    backend = "none"
    if use_features:
        # ⚠️ 这里**不允许静默降级**（2026-10-04 实测踩过）：
        #   原实现 `except Exception: print(警告)` 把 AssertionError 吞掉，
        #   导致「特征命中 0 对」看起来像「没有语义重复」，实际是**精筛根本没跑**。
        #   而精筛是方法论核心 —— 跑不了就必须让整个流程失败。
        if verbose:
            print(f"[dedup] 提取 DINOv2 特征（reuse_cache={reuse_cache}）...")
        feats, backend = _get_features(paths, device=device,
                                       reuse_cache=reuse_cache, verbose=verbose)
        if len(feats) != n:
            raise GuardError(f"特征数 {len(feats)} != 图片数 {n}（缓存与数据不一致）")
        # 语义级重复：不同角度拍的同一实物（哈希抓不到，但本方法已不用哈希）
        sim = feats @ feats.T
        ii, jj = np.triu_indices(n, k=1)
        hit = sim[ii, jj] >= feature_threshold
        for a, b in zip(ii[hit].tolist(), jj[hit].tolist()):
            uf.union(a, b)
            n_ft += 1
        if verbose:
            print(f"[dedup]余弦精筛union {n_ft} 对 "
                  f"(backend={backend}, cos≥{feature_threshold})")
        if n_ft == 0:
            # 前缀已保证至少 161 对，这里为 0 意味着特征提取有问题 → 必须炸
            raise GuardError(
                f"余弦 ≥{feature_threshold} 命中 0 对。DC-02R 已知存在 80+ 组"
                f"`_aug` 近重复，**0 命中说明特征提取有问题**（缓存损坏 / "
                f"路径错位 / 阈值不匹配），必须先排查再跑折划分。")

    # ---------------------------------------------- 3) 连通分量 → group_id
    gmap = {}
    gids = []
    for i in range(n):
        r = uf.find(i)
        if r not in gmap:
            gmap[r] = len(gmap)
        gids.append(gmap[r])

    members = defaultdict(list)
    for i, g in enumerate(gids):
        members[g].append(i)
    multi = {g: [paths[i] for i in idx]
             for g, idx in members.items() if len(idx) > 1}

    # 🔴 跨类污染检查：一个 group 里出现多个类别 = 把不同类误判成同一实物。
    #   若出现，只能靠**拆分**而非 union 修：group-aware 折划分要求 group 纯类，
    #   否则 StratifiedGroupKFold 直接失效。
    n_impure = 0
    for g, idx in members.items():
        if len({labels[i] for i in idx}) > 1:
            n_impure += 1
    if n_impure:
        ex = []
        for g, idx in members.items():
            if len({labels[i] for i in idx}) > 1:
                ex.append({"group": g, "n": len(idx),
                           "labels": sorted({labels[i] for i in idx}),
                           "files": [Path(paths[i]).name for i in idx[:4]]})
            if len(ex) >= 5:
                break
        raise GuardError(
            f"检测到 {n_impure} 个跨类污染 group（示例 {json.dumps(ex, ensure_ascii=False)}）。"
            f"阈值过松或前缀规则有误——**必须先修阈值再跑折划分**，"
            f"否则 group-aware 口径失效。")

    # 🔴 防线9：前缀 ground truth 召回率必须 100%。
    #   「跨类污染 0」只证明**没误合并**，不证明**没漏合并** —— 后者会让去泄漏
    #   打折，指标重新虚高。DC-02R 自带 `xxx`/`xxx_aug`/`xxx_original` 是唯一
    #   确定性 ground truth，任何一组被拆开都说明聚组逻辑坏了。
    gt_groups = defaultdict(list)
    for i, p in enumerate(paths):
        gt_groups[_prefix_key(p)].append(i)
    n_gt_multi = sum(1 for v in gt_groups.values() if len(v) > 1)
    n_gt_intact = sum(1 for v in gt_groups.values()
                      if len(v) > 1 and len({gids[i] for i in v}) == 1)
    if n_gt_multi and n_gt_intact != n_gt_multi:
        bad = [k for k, v in gt_groups.items()
               if len(v) > 1 and len({gids[i] for i in v}) > 1]
        raise GuardError(
            f"前缀 ground truth 召回 {n_gt_intact}/{n_gt_multi} "
            f"（<100%）。被拆开的前缀组: {bad[:5]}。"
            f"说明聚组逻辑漏掉了**确定性**真重复 → 去泄漏不彻底，指标会虚高。")
    if verbose:
        print(f"[dedup] ✅ 防线9 前缀 ground truth 召回 "
              f"{n_gt_intact}/{n_gt_multi} = 100%")

    return DedupResult(paths=paths, labels=list(labels),
                       group_ids=gids, n_groups=len(gmap),
                       n_dup_pairs_phash=n_ph_diag, n_dup_pairs_feature=n_ft,
                       n_dup_pairs_prefix=n_prefix,
                       feature_threshold=feature_threshold,
                       phash_threshold=phash_diag_threshold, backend=backend,
                       multi_groups=multi, n_impure_groups=n_impure,
                       max_group=max(len(v) for v in members.values()))


def _get_features(paths, device: str = "cuda", reuse_cache: bool = True,
                  verbose: bool = True):
    """取 DINOv2 特征 [N,768]，带磁盘缓存。

    ⚠️ 两个已实测的坑：
    1) **逐张 vs 分批**：`np.stack([fx(im) for im in imgs])` 逐张推理 1,912 张
       约 5 min；`embed_batch`（batch=16, fp16）约 40 s。**必须分批**。
    2) **缓存必须预检可写**：先写探针文件再提特征，否则 5 min 算完才发现
       np.save 失败（原生 Windows Python 认不了 Git Bash 的 `/tmp`）。
    """
    if reuse_cache and FEAT_CACHE.exists() and PATH_CACHE.exists():
        cached = [str(x) for x in np.load(PATH_CACHE, allow_pickle=True)]
        if cached == list(paths):
            F = np.load(FEAT_CACHE)
            if len(F) == len(paths):
                if verbose:
                    print(f"[dedup] 复用特征缓存 {F.shape} → {FEAT_CACHE}")
                return F, "cache"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    probe = CACHE_DIR / "_write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    fx = FeatureExtractor(device=device)
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB").resize(fx.input_size))
    F = fx.embed_batch(imgs, batch=16)
    del imgs
    np.save(FEAT_CACHE, F)
    np.save(PATH_CACHE, np.array([str(p) for p in paths], dtype=object))
    if verbose:
        print(f"[dedup] 特征已缓存 {F.shape} → {FEAT_CACHE}")
    return F, fx.backend


# ---------------------------------------------------------------- 折划分
def make_folds(labels, group_ids, n_splits: int = N_FOLDS, seed: int = SEED):
    """产出两套划分：普通分层 / 去泄漏 group-aware。"""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    y = np.asarray(labels)
    g = np.asarray(group_ids)

    strat = {}
    for fold, (_, te) in enumerate(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)):
        for i in te:
            strat[i] = fold

    grouped = {}
    for fold, (_, te) in enumerate(
            StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
            .split(y, y, groups=g)):
        for i in te:
            grouped[i] = fold

    # 🔴 防线10：group-aware 划分必须真的「一个 group 只落一折」。
    #   若某 group 跨折，StratifiedGroupKFold 就退化成普通分层，
    #   而报告里的「去泄漏口径」是假的 —— **静默失效，必须当场炸**。
    g2f = defaultdict(set)
    for i, fold in grouped.items():
        g2f[group_ids[i]].add(fold)
    bad = {g: sorted(fs) for g, fs in g2f.items() if len(fs) > 1}
    if bad:
        raise GuardError(
            f"group-aware 划分失效：{len(bad)} 个 group 仍跨折"
            f"（示例 {list(bad.items())[:3]}）。"
            f"通常因少数类样本太少（Off_platform 仅 91 张）无法满足 "
            f"n_splits={n_splits}，需减少折数或改用 GroupKFold。")
    return strat, grouped


def leakage_estimate(strat, grouped, group_ids):
    """量化「普通分层把同一实物的图拆到不同折」的程度 = **泄漏的直接度量**。

    🔴 2026-10-04 修正：原实现统计「每折包含几个 group」，>1 就算跨折。
       但任何折都必然含多个 group（1,912 张 / 5 折），所以这个指标**恒为 5/5**，
       零信息量。正确的度量是**从 group 侧看**：
          有多少个 group 的成员被拆散到了 ≥2 个折（这些就是泄漏的图）
           泄漏张数 = 这些 group 里落在测试折的图数
    """
    from collections import defaultdict

    # ⚠️ 2026-10-04 实测bug：`enumerate(strat.values())` 的下标**不是原图索引**。
    #   strat 是 `for i in te: strat[i] = fold` 填充的 dict，插入顺序按 fold 轮次
    #   而非 i 升序 → 曾经算出「泄漏 0 张 / 0 group」的荒谬结果。
    #   **必须遍历 items() 拿真实索引。**
    g2folds = defaultdict(set)
    for i, f in strat.items():
        g2folds[group_ids[i]].add(f)
    split_groups = {g: fs for g, fs in g2folds.items() if len(fs) > 1}

    n_multi = defaultdict(int)
    for i in strat:
        n_multi[group_ids[i]] += 1
    n_multi_groups = sum(1 for v in n_multi.values() if v > 1)
    n_leaked_images = sum(1 for i in strat if len(g2folds[group_ids[i]]) > 1)

    # 🔴 防线11：普通分层必然制造泄漏。163 张折叠图在 5 折下期望泄漏约
    #   163×(4/5) ≈ 130 张。若算出 0，说明上面的索引又错了 —— 当场炸，别把
    #   「度量失效」当成「没有泄漏」写进报告。
    n_folded = len(strat) - len(g2folds)
    if n_multi_groups > 0 and n_leaked_images == 0:
        raise GuardError(
            f"泄漏度量恒为 0 不合理：有 {n_multi_groups} 个多图 group，"
            f"5 折普通分层下必然把它们拆散。请检查 leakage_estimate 的索引口径。")
    return {
        "n_groups": len(g2folds),
        "n_multi_image_groups": n_multi_groups,
        "n_images_folded": n_folded,
        "n_groups_split_across_folds": len(split_groups),
        "n_images_in_split_groups": n_leaked_images,
        "pct_images_leaked": round(100.0 * n_leaked_images / len(strat), 2),
        "note": ("普通分层下被拆散到多折的 group 数/涉及张数 = 泄漏的直接上界。"
                 "泄漏幅度 = 两套划分的 macro-F1 之差（需训练后才知道，"
                 "本项只说明'有没有泄漏'和'泄漏多大范围'）。"),
    }


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="data/p1_folds.json")
    ap.add_argument("--no-features", action="store_true",
                    help="只用文件名前缀（不推荐：会漏掉不同视角拍的同一实物）")
    ap.add_argument("--no-cache", action="store_true",
                    help="强制重提 DINOv2 特征（默认复用 exp/_cache 缓存）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from .guards import probe_dataset
    probe = probe_dataset(args.data_root)
    root = probe.root
    print(f"[dedup] 数据根: {root}")
    print(f"[dedup] 基准张数: {EXPECTED_COUNTS} 期望合计 {EXPECTED_TOTAL}")

    paths, labels = [], []
    for c, n in EXPECTED_COUNTS.items():
        fs = sorted((root / c).iterdir())
        fs = [f for f in fs if f.suffix.lower() in IMG_EXT]
        if len(fs) != n:
            print(f"[dedup] ⚠️ {c}: 实测 {len(fs)} != 基准 {n}（继续，但报告会注明）")
        paths += fs
        labels += [c] * len(fs)
    print(f"[dedup] 待处理 {len(paths)} 张（期望 {EXPECTED_TOTAL}）")

    res = build_groups(paths, labels,
                       use_features=not args.no_features,
                       device=args.device,
                       reuse_cache=not args.no_cache)
    print("\n" + res.summary())

    strat, grouped = make_folds(res.labels, res.group_ids)
    leak = leakage_estimate(strat, grouped, res.group_ids)
    print(f"\n[fold] 泄漏直接度量: {leak['n_groups_split_across_folds']}/"
          f"{leak['n_groups']} 个 group 被普通分层拆散到多折，"
          f"涉及 {leak['n_images_in_split_groups']} 张"
          f"（{leak['pct_images_leaked']}%）")

    out = {
        "seed": SEED, "n_folds": N_FOLDS,
        "data_root": str(root),
        "n_images": len(res.paths),
        "n_groups": res.n_groups,
        "n_dup_pairs_prefix": res.n_dup_pairs_prefix,
        "n_dup_pairs_feature": res.n_dup_pairs_feature,
        "n_dup_pairs_phash_diag_only": res.n_dup_pairs_phash,
        "phash_not_used_for_union": True,
        "feature_threshold": res.feature_threshold,
        "feature_backend": res.backend,
        "max_group": res.max_group,
        "n_impure_groups": res.n_impure_groups,
        "expected_counts": EXPECTED_COUNTS,
        "leakage_clue": leak,
        "images": [
            {"path": p, "label": lb, "group": g,
             "fold_stratified": strat[i], "fold_grouped": grouped[i]}
            for i, (p, lb, g) in enumerate(zip(res.paths, res.labels, res.group_ids))
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n[产出] {args.out}（两套折划分 + group_id，可复现）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
