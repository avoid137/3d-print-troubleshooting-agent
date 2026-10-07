"""P1 · OOD 域检测器（域外安全阀）。

为什么需要它
------------
P1 视觉的分类精度只在源域成立：DC-02R 内折外 macro-F1 **99.73%**，
换机器（DC-04）**6.25%**，低于 5 类随机 20%，形态是「输出塌缩」
（86.6% 的图，含正常件，都被自信地判成 Stringing）。

工程处置不是「追求跨任意域都准」（那是学术难题），而是：

    域受控（固定拍摄条件）+ 域外安全阀（不认识就降级/报警）

这与 P0 的「库外问题→拒答」是同一套哲学——**知道自己不知道，比答得准更重要**。

🔴 已证伪的路线（别再试）
-------------------------
`exp/d11_precheck_d8_scores.py` 用 d8 的 1200 条域外样本 + D3 域内 coverage 曲线
实测：**max-softmax 做 OOD，近似 AUROC = 0.4136（< 0.5，比随机还差）**。
域外 median 0.980 vs 域内 0.99989，几乎重叠；域外三类（缺陷/正常）分数分布
几乎相同 → **置信度不携带域信息**。所以：

    ❌ max-softmax / margin / energy 等「置信度类」分数 —— 一律不用作主判据
    ✅ 特征空间距离（cosine-to-centroid / kNN / Mahalanobis）—— 有强信号：
       infer.py 已测 DC-04 三类质心与 DC-02R 全体质心余弦仅 0.53~0.55。

本模块实现后者。所有分数统一方向：**越大越像 OOD**。
阈值用**域内样本**分位数标定（conformal 风格），保证域内误报率可控。

不依赖 torch：特征由外部传入（4060 上由 `infer.embed_paths` 提取）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["OODDetector", "auroc", "tpr_at_fpr", "CONFIDENCE_METHODS", "FEATURE_METHODS"]

#: 置信度类分数（已证伪，仅保留作对照/反例，**不要用作主判据**）
CONFIDENCE_METHODS = ("max_softmax", "margin", "energy")
#: 特征距离类分数（推荐）
FEATURE_METHODS = ("cosine_centroid", "knn", "mahalanobis")


def _l2norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


@dataclass
class OODDetector:
    """多分数域外检测器。特征形态 [N, D]（本项目的 D=1280，efficientnet_b0 冻结特征）。

    用法（4060 上）::

        det = OODDetector(methods=("cosine_centroid", "knn", "mahalanobis"), k=10)
        det.fit(F_id, y_id)                    # 只用**域内**特征 + 标签
        s_id  = det.score(F_id)                # 域内分数（标定用）
        s_ood = det.score(F_ood)               # 域外分数（评测用）
        for m in det.methods:
            print(m, "AUROC", auroc(s_ood[m], s_id[m]))
        thr = det.calibrate(s_id, alpha=0.05)  # 域内误报 <=5% 的阈值
        print(det.workpoint(s_ood, thr))       # 该阈值下域外检出率
    """

    methods: tuple = FEATURE_METHODS
    k: int = 10
    #: 特征是否先做 L2 归一化（余弦/kNN 用；Mahalanobis 用原始尺度）
    normalize: bool = True

    # --- 拟合产物 ---
    _centroids: np.ndarray | None = field(default=None, repr=False)  # [C, D] 归一化类质心
    _X_id: np.ndarray | None = field(default=None, repr=False)       # [N, D] 域内特征（归一化）
    _mu: np.ndarray | None = field(default=None, repr=False)         # Mahalanobis 均值
    _pinv: np.ndarray | None = field(default=None, repr=False)       # 协方差伪逆

    # ---------------------------------------------------------------- 拟合
    def fit(self, X_id: np.ndarray, y_id: np.ndarray) -> "OODDetector":
        """只用域内特征 + 标签拟合。**严禁混入域外样本**（否则阈值失真）。"""
        X_id = np.asarray(X_id, dtype=np.float64)
        y_id = np.asarray(y_id)
        if X_id.ndim != 2:
            raise ValueError(f"X_id 需为 [N, D]，得到 {X_id.shape}")
        if len(X_id) != len(y_id):
            raise ValueError(f"X/y 数目不一致: {len(X_id)} vs {len(y_id)}")
        if len(X_id) < 10:
            raise ValueError("域内样本过少（<10），OOD 标定不可靠")

        Xn = _l2norm(X_id) if self.normalize else X_id
        classes = np.unique(y_id)
        self._centroids = np.stack([_l2norm(Xn[y_id == c].mean(0, keepdims=True))[0]
                                    for c in classes])
        self._X_id = Xn

        # 全局协方差（共享）→ Mahalanobis
        self._mu = X_id.mean(0)
        Xc = X_id - self._mu
        cov = (Xc.T @ Xc) / max(len(X_id) - 1, 1)
        # 加白噪声保证可逆（高维 + 样本有限时协方差奇异）
        cov += np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0]) * 1e-3
        self._pinv = np.linalg.pinv(cov)
        return self

    # ---------------------------------------------------------------- 打分
    def score(self, X: np.ndarray) -> dict:
        """返回 {方法名: 分数数组}，**统一方向：越大越像 OOD**。"""
        if self._centroids is None:
            raise RuntimeError("请先 fit()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xn = _l2norm(X) if self.normalize else X
        out: dict[str, np.ndarray] = {}

        # --- 特征距离族（推荐） ---
        if "cosine_centroid" in self.methods:
            # 与最近类质心的最大余弦；OOD = 1 - max_cos
            cos = Xn @ self._centroids.T                      # [N, C]
            out["cosine_centroid"] = 1.0 - cos.max(axis=1)

        if "knn" in self.methods:
            # 到域内 k 近邻的平均**角距离**（归一化特征下 = 1 - 余弦）
            sim = Xn @ self._X_id.T                           # [N, N_id]
            k = min(self.k, sim.shape[1])
            part = np.partition(sim, -k, axis=1)[:, -k:]
            out["knn"] = 1.0 - part.mean(axis=1)

        if "mahalanobis" in self.methods:
            d = X - self._mu
            out["mahalanobis"] = np.sqrt(np.maximum(
                np.einsum("ij,jk,ik->i", d, self._pinv, d), 0.0))

        # --- 置信度族（已证伪，仅作对照；需要概率输入，见 score_proba） ---
        return out

    @staticmethod
    def score_proba(proba: np.ndarray) -> dict:
        """置信度类分数（**已证伪，仅供对照**）。proba=[N, C] 概率矩阵。

        方向同样统一为「越大越 OOD」：max-softmax 用 1-p（低置信 → 大分数）。
        """
        proba = np.asarray(proba, dtype=np.float64)
        out: dict[str, np.ndarray] = {}
        srt = np.sort(proba, axis=1)
        out["max_softmax"] = 1.0 - srt[:, -1]
        out["margin"] = 1.0 - (srt[:, -1] - srt[:, -2]) if proba.shape[1] >= 2 \
            else np.zeros(len(proba))
        return out

    # ---------------------------------------------------------------- 标定
    def calibrate(self, scores_id: dict, alpha: float = 0.05) -> dict:
        """用**域内**分数分位数定阈值 → 域内误报率 ≈ alpha（conformal 风格）。"""
        thr = {}
        for m, s in scores_id.items():
            thr[m] = float(np.quantile(np.asarray(s), 1.0 - alpha))
        return thr

    @staticmethod
    def workpoint(scores_ood: dict, thresholds: dict) -> dict:
        """给定阈值，报「域外检出率」（= 分数 > 阈值 的比例）。"""
        return {m: round(float((scores_ood[m] > thresholds[m]).mean()), 4)
                for m in thresholds if m in scores_ood}

    @staticmethod
    def decide(scores: dict, thresholds: dict) -> dict:
        """逐样本判定：任一方法越界即 OOD（也可改成投票）。"""
        n = len(next(iter(scores.values())))
        flags = {m: np.asarray(scores[m]) > thresholds[m] for m in thresholds if m in scores}
        any_flag = np.zeros(n, dtype=bool)
        for f in flags.values():
            any_flag |= f
        return {"is_ood": any_flag, "n_ood": int(any_flag.sum()),
                "rate": float(any_flag.mean()), "per_method": flags}


# -------------------------------------------------------------------- 评测
def auroc(scores_ood: np.ndarray, scores_id: np.ndarray) -> float:
    """AUROC（越大越 OOD 的方向下）。0.5=无区分度；<0.5 表示方向反了。"""
    a = np.asarray(scores_ood, dtype=np.float64)
    b = np.asarray(scores_id, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # 秩和法（Mann-Whitney U）
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列：取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        avg = sums / cnt
        ranks = avg[inv]
    r_ood = ranks[:len(a)].sum()
    u = r_ood - len(a) * (len(a) + 1) / 2
    return float(u / (len(a) * len(b)))


def tpr_at_fpr(scores_ood: np.ndarray, scores_id: np.ndarray, fpr: float) -> float:
    """限定域内误报 <= fpr 时，域外检出率（TPR）。"""
    thr = float(np.quantile(np.asarray(scores_id), 1.0 - fpr))
    return float((np.asarray(scores_ood) > thr).mean())


# -------------------------------------------------------------------- 自测
def _self_test() -> int:
    """合成数据自测：域内两团 + 域外一团，特征距离分数应能分开；置信度分数应失败。"""
    rng = np.random.default_rng(0)
    D = 64
    c0 = np.zeros(D); c0[0] = 1.0
    c1 = np.zeros(D); c1[1] = 1.0
    X_id = np.vstack([c0 + rng.normal(0, 0.05, (200, D)),
                      c1 + rng.normal(0, 0.05, (200, D))])
    y_id = np.array([0] * 200 + [1] * 200)
    # 域外：正交方向，且**高置信**（模拟「域外也自信」）
    c_ood = np.zeros(D); c_ood[2] = 1.0
    X_ood = c_ood + rng.normal(0, 0.05, (200, D))

    det = OODDetector(methods=("cosine_centroid", "knn", "mahalanobis"), k=10).fit(X_id, y_id)
    s_id, s_ood = det.score(X_id), det.score(X_ood)
    ok = True
    for m in det.methods:
        a = auroc(s_ood[m], s_id[m])
        status = "OK" if a > 0.95 else "FAIL"
        if a <= 0.95:
            ok = False
        print(f"[selftest] 特征分数 {m:18s} AUROC={a:.4f}  {status}")

    # 置信度分数：构造「域外概率也很高」→ AUROC 应接近 0.5 或更差
    p_id = np.eye(3)[np.zeros(len(X_id), dtype=int)]  # 无关，仅示意接口
    s2 = OODDetector.score_proba(np.vstack([
        np.tile([0.98, 0.01, 0.01], (200, 1)),
        np.tile([0.98, 0.01, 0.01], (200, 1))]))
    print(f"[selftest] 置信度接口 max_softmax 形状={s2['max_softmax'].shape} (方向=越大越OOD)")

    # 标定 + 工作点
    thr = det.calibrate(s_id, alpha=0.05)
    wp = det.workpoint(s_ood, thr)
    print(f"[selftest] 域内误报<=5% 阈值={ {k: round(v,4) for k,v in thr.items()} }")
    print(f"[selftest] 对应域外检出率={wp}")
    if min(wp.values()) < 0.9:
        ok = False
        print("[selftest] FAIL: 工作点检出率应 >0.9")
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
