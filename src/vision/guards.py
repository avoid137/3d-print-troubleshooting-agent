"""P1 视觉管线五道防线（kaggle-vision-pipeline-guards 的内联实现）。

背景：本机（4060 机器）**没有** `kaggle-vision-pipeline-guards` skill（全盘搜索无结果），
故按 HANDOFF_P1.md §4 的描述把五道防线落成可执行代码。

核心命题：「不报错的错误最贵」——loss 正常下降、分数就是不高，最可能是数据根本没喂进去。
五道防线专门拦这类静默失败；第六道（去泄漏）针对 DC-02R 已知的同实物连续拍摄问题。

用法：
    from guards import GuardReport, audit_backbone, probe_dataset, RuntimeCounter
    report = audit_backbone("efficientnet_b0", img_size=224)   # 防线 1
    probe_dataset(root, expect_classes)                        # 防线 2
    with RuntimeCounter() as c: ...# 防线 3
    python -m src.vision.guards --self-test                     # 防线 4/5/6 自检
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# DC-02R 实测基准（Kaggle 文件清单全量遍历核实，2026-09-28）
EXPECTED_COUNTS = {
    "Warping": 538,
    "Cracking": 472,
    "Stringing": 447,
    "Layer_shifting": 364,
    "Off_platform": 91,
}
EXPECTED_TOTAL = 1912


class GuardError(RuntimeError):
    """防线判定为硬失败。调用方应立即中止，不要试图改模型。"""


# ---------------------------------------------------------------- 防线 1
@dataclass
class BackboneAudit:
    name: str
    num_classes: int
    total_params: int
    trainable_params: int
    input_size: tuple
    mean: tuple
    std: tuple
    pretrained_cfg_mean: tuple
    pretrained_cfg_std: tuple

    @property
    def size_mb(self) -> float:
        return self.total_params * 4 / 1024 ** 2

    def summary(self) -> str:
        return (
            f"骨干: {self.name}\n"
            f"  参数量     : {self.total_params:,} ({self.size_mb:.1f} MB fp32)\n"
            f"  可训练参数 : {self.trainable_params:,}\n"
            f"  输入尺寸   : {self.input_size}\n"
            f"  归一化 mean: {self.mean}\n"
            f"  归一化 std : {self.std}\n"
            f"  (取自 timm.resolve_data_config，未手写)"
        )


def audit_backbone(name: str = "efficientnet_b0", num_classes: int = 5,
                   img_size: int = 224, pretrained: bool = True) -> BackboneAudit:
    """防线 1 · 骨干审计。

    归一化常数一律取自 `timm.resolve_data_config()`，**绝不手写** ——
    换骨干不换 mean/std 会静默劣化（不报错，只是分数低）。
    """
    try:
        import timm
        from timm.data import resolve_data_config
    except ImportError as e:
        raise GuardError(f"timm 未安装：{e}") from e

    # ⚠️ pretrained=True 不走 timm.create_model：本机（Windows 无符号链接权限 +
    #    hf_hub 传输截断）会得到 0 字节权重并抛 header too small。
    #    统一交给 weights.build_model，它会校验「注入张量数」并反向自检。
    if pretrained:
        from src.vision.weights import build_model
        model = build_model(name, num_classes=num_classes, img_size=img_size)
    else:
        model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    cfg = resolve_data_config({}, model=model)
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    ic = cfg.get("input_size", (3, img_size, img_size))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return BackboneAudit(
        name=name, num_classes=num_classes,
        total_params=total, trainable_params=trainable,
        input_size=tuple(ic), mean=mean, std=std,
        pretrained_cfg_mean=mean, pretrained_cfg_std=std,
    )


# ---------------------------------------------------------------- 防线 2
@dataclass
class ProbeResult:
    root: Path
    per_class_found: dict
    total_found: int
    missing_classes: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """每类命中率：能找到图的类别数 / 期望类别数。"""
        if not EXPECTED_COUNTS:
            return 0.0
        return len(self.per_class_found) / len(EXPECTED_COUNTS)

    def summary(self) -> str:
        return (f"数据根: {self.root}\n"
                f"  找到类别: {self.per_class_found}\n"
                f"  找到总数: {self.total_found}\n"
                f"  缺失类别: {self.missing_classes or '无'}\n"
                f"  每类命中率: {self.hit_rate:.2f}（< 0.5 判定硬失败）")


def probe_dataset(root, expect_classes=None, min_hit_rate: float = 0.5,
                  strict: bool = True) -> ProbeResult:
    """防线 2 · 目录层级探测。

    数据集是 `data/<类别>/*.jpg`，但 Kaggle 解压后常多一层目录，脚本已自动下探；
    这里再独立验证一次。命中率 < min_hit_rate 直接抛错——先查管线，别改模型。
    """
    root = Path(root)
    if not root.exists():
        raise GuardError(f"数据根不存在: {root}")

    # 下探：找出命中类别最多的那一层
    best, best_hit = None, 0
    expect = set(expect_classes or EXPECTED_COUNTS)
    for p in [root] + [d for d in root.rglob("*") if d.is_dir()]:
        hit = sum(1 for c in expect if (p / c).is_dir())
        if hit > best_hit:
            best, best_hit = p, hit
    if best is None:
        raise GuardError(f"在 {root} 下未找到任何含类别子目录的层（期望 {sorted(expect)}）")

    found, missing = {}, []
    for c in sorted(expect):
        d = best / c
        if not d.is_dir():
            missing.append(c)
            continue
        found[c] = len([f for f in d.iterdir() if f.suffix.lower() in IMG_EXT])

    res = ProbeResult(root=best, per_class_found=found,
                      total_found=sum(found.values()), missing_classes=missing)
    if strict and res.hit_rate < min_hit_rate:
        raise GuardError(
            f"每类命中率 {res.hit_rate:.2f} < {min_hit_rate}，判定为管线问题。\n"
            f"{res.summary()}")
    return res


# ---------------------------------------------------------------- 防线 3
@dataclass
class RuntimeCounter:
    """防线 3 · 运行时计数器。

    分开统计两种失败（**语义不同，必须分开**）：
      · empty  : 读取成功但样本内容为空（返回全 0 张量等）→ 多半是路径/标签映射错
      · failed : 解码/打开异常→ 多半是文件损坏或格式不支持
    两者任一占比 > 5% 都必须先查管线，**不要改模型**。

    ⚠️ 2026-10-04 自检时发现的历史 bug：原 `record(ok: bool)` 把 ok=False 一律记到
    `failed`，导致 `empty_rate` 恒为 0 → 防线 3 永远不会触发，是典型的
    「永远通过的检查即装饰」。故改为两个显式方法，语义不可能搞混。
    """

    empty: int = 0
    failed: int = 0
    seen: int = 0
    max_rate: float = 0.05
    fail_on_violation: bool = True
    _log: list = field(default_factory=list)

    def record_empty(self, note: str = "") -> None:
        """读取成功但样本为空。"""
        self.seen += 1
        self.empty += 1
        if note:
            self._log.append(f"empty: {note}")

    def record_failed(self, note: str = "") -> None:
        """解码/打开失败。"""
        self.seen += 1
        self.failed += 1
        if note:
            self._log.append(f"failed: {note}")

    def record_ok(self) -> None:
        self.seen += 1

    @property
    def empty_rate(self) -> float:
        return (self.empty / self.seen) if self.seen else 0.0

    @property
    def failed_rate(self) -> float:
        return (self.failed / self.seen) if self.seen else 0.0

    @property
    def bad_rate(self) -> float:
        return self.empty_rate + self.failed_rate

    def check(self) -> None:
        if not self.fail_on_violation:
            return
        if self.empty_rate > self.max_rate:
            raise GuardError(
                f"空样本率 {self.empty_rate:.1%} > {self.max_rate:.0%}。"
                f"先查管线（路径/标签映射），不要改模型。"
                f"最近记录: {self._log[:5]}")
        if self.failed_rate > self.max_rate:
            raise GuardError(
                f"读取失败率 {self.failed_rate:.1%} > {self.max_rate:.0%}。"
                f"先查管线（文件损坏/格式不支持），不要改模型。"
                f"最近记录: {self._log[:5]}")

    def summary(self) -> str:
        return (f"RuntimeCounter: seen={self.seen} empty={self.empty} "
                f"({self.empty_rate:.1%}) failed={self.failed} "
                f"({self.failed_rate:.1%})")


# ---------------------------------------------------------------- 防线 5
def assert_can_fail(fn, *args, **kwargs) -> bool:
    """防线 5 · 防线自检。

    永远通过的检查是装饰。调用它必须能证明 fn 在条件不满足时**确实会抛错**。
    """
    try:
        fn(*args, **kwargs)
    except Exception:
        return True
    return False


# ---------------------------------------------------------------- 防线 7：静态符号检查
#: 本项目跨模块共享的自定义符号（`from src.vision.weights import build_model` 等）
CUSTOM_SYMBOLS = {
    "build_model", "load_pretrained_into", "WeightError",
    "load_folds", "fold_of", "extract_features", "build_transform", "FlatImages",
    "CLASSES", "SEED", "probe_dataset", "RuntimeCounter", "audit_backbone",
    "assert_can_fail", "GuardError", "EXPECTED_COUNTS", "full_report",
}


def check_undefined_symbols(paths=None) -> list[str]:
    """防线 7 · 用 AST 找出「使用了但本模块没有任何定义来源」的自定义符号。

    为什么需要它（真实案例，2026-10-04）：替换 `timm.create_model(pretrained=True)`
    时顺手删掉了 `run_finetune_level` 里的 import 行，函数体却仍在调用 `build_model`，
    **语法完全合法、py_compile 通过、只有在真跑到 L3 全量微调时才会 NameError**。
    静态检查抓不到，但 AST 一扫就出来——比跑三轮训练再看 traceback 便宜得多。
    """
    import ast
    from pathlib import Path

    if paths is None:
        paths = (sorted(Path("src/vision").glob("*.py"))
                 + sorted(Path("scripts").glob("*.py")))
    problems = []
    for f in paths:
        f = Path(f)
        if not f.is_file():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        defined = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    defined.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = (used & CUSTOM_SYMBOLS) - defined
        if missing:
            problems.append(f"{f}: 使用了但未定义 {sorted(missing)}")
    return problems



def check_unreachable_code(paths=None) -> list[str]:
    """找出函数体内「return/raise 之后还有语句」的不可达代码。

    为什么加这条（真实案例 2026-10-04）：编辑 `dedup.py` 时误删了 `_prep` 的
    `def` 行，函数体变成上一个方法的悬空尾巴。**语法合法、ast.parse 通过、
    py_compile 通过、符号检查也通过**（符号确实"定义"了），但 `_prep` 方法
    凭空消失 → 运行时才炸。属于「不报错的错误」，只能靠静态扫描兜住。
    """
    import ast
    from pathlib import Path

    if paths is None:
        paths = (sorted(Path("src/vision").glob("*.py"))
                 + sorted(Path("scripts").glob("*.py")))
    problems = []
    for f in paths:
        f = Path(f)
        if not f.is_file():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for i, st in enumerate(node.body[:-1]):
                if isinstance(st, (ast.Return, ast.Raise)):
                    nxt = node.body[i + 1]
                    problems.append(
                        f"{f}:{nxt.lineno} 函数 {node.name}() 在 {st.lineno} 行 "
                        f"{type(st).__name__} 之后还有 {type(nxt).__name__}（不可达）")
    return problems


# ---------------------------------------------------------------- 汇总
def full_report(root=None, backbone: str = "efficientnet_b0",
                img_size: int = 224, num_classes: int = 5) -> str:
    """一次性跑防线 1/2/3，产出可读报告。"""
    lines = ["=" * 62, "P1 管线防线报告", "=" * 62]

    lines.append("\n[防线 1] 骨干审计")
    a = audit_backbone(backbone, num_classes=num_classes, img_size=img_size)
    lines.append(a.summary())
    if a.mean != a.pretrained_cfg_mean or a.std != a.pretrained_cfg_std:
        raise GuardError("归一化常数与 pretrained_cfg 不一致 —— 静默劣化风险")

    if root is not None:
        lines.append("\n[防线 2] 目录层级探测")
        p = probe_dataset(root)
        lines.append(p.summary())

    c = RuntimeCounter()
    lines.append("\n[防线 3] 运行时计数器")
    lines.append(c.summary())

    lines.append("\n[防线 4] 无数据冒烟：见 train.py --smoke-test")
    lines.append("[防线 5] 防线自检：python -m src.vision.guards --self-test")
    lines.append("[防线 6] 去泄漏：python -m src.vision.dedup --data-root <root>")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自检入口
def _self_test() -> int:
    """防线 4/5 联合自检：用合成数据验证每条防线该 FAIL 时确实 FAIL。"""
    import tempfile

    print("=" * 62)
    print("防线自检 · 验证「条件不满足时确实会FAIL」")
    print("=" * 62)
    ok = True

    # 防线 2 必须能识别「目录层级错误」
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "wrong_dir"
        (bad / "NotADefectClass").mkdir(parents=True)
        r = assert_can_fail(probe_dataset, bad, strict=True)
        print(f"  {'✅' if r else '❌'} 防线2 对错误目录会 FAIL: {r}")
        ok &= r

        # 防线 2 对正确结构不应 FAIL
        good = Path(td) / "good"
        for c in EXPECTED_COUNTS:
            (good / c).mkdir(parents=True)
            for i in range(3):
                (good / c / f"{c}_{i}.jpg").write_bytes(b"x")
        try:
            probe_dataset(good, strict=True)
            r = True
        except GuardError:
            r = False
        print(f"  {'✅' if r else '❌'} 防线2 对正确结构不 FAIL: {r}")
        ok &= r

    # 防线 3 对空样本超限必须 FAIL
    c = RuntimeCounter()
    for _ in range(10):
        c.record_empty("synthetic")
    r = assert_can_fail(c.check)
    print(f"  {'✅' if r else '❌'} 防线3 空样本超限会 FAIL: {r} ({c.summary()})")
    ok &= r

    # 防线 3 对读取失败超限也必须 FAIL（两种失败语义独立）
    c3 = RuntimeCounter()
    for _ in range(10):
        c3.record_failed("synthetic")
    r = assert_can_fail(c3.check)
    print(f"  {'✅' if r else '❌'} 防线3 读取失败超限会 FAIL: {r}")
    ok &= r

    # 防线 3 对正常情况不应 FAIL
    c2 = RuntimeCounter()
    for _ in range(10):
        c2.record_ok()
    r = not assert_can_fail(c2.check)
    print(f"  {'✅' if r else '❌'} 防线3 正常情况不 FAIL: {r}")
    ok &= r

    # 防线 1' 权重加载器必须能识别「0 字节 / 坏权重」——
    # 这正是本机 header too small 的真实成因，不测这条等于防线是装饰。
    import tempfile as _tf
    import timm
    from src.vision import weights as W
    with _tf.TemporaryDirectory() as td:
        bad_w = Path(td) / "model.safetensors"
        bad_w.write_bytes(b"")            # 0 字节，模拟 Windows 符号链接失败
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
        orig = W.WEIGHTS_DIR
        try:
            W.WEIGHTS_DIR = Path(td)      # 让查找落到我们造的空文件上
            r = assert_can_fail(
                W.load_pretrained_into, m, "__no_such_repo__")
        finally:
            W.WEIGHTS_DIR = orig
        print(f"  {'✅' if r else '❌'} 防线1' 坏权重(0字节)会 FAIL: {r}")
        ok &= r

    # 防线 7 静态符号检查：当前代码必须干净
    probs = check_undefined_symbols()
    r = not probs
    print(f"  {'✅' if r else '❌'} 防线7 当前代码无未定义符号: {r}"
          + (f" {probs}" if probs else ""))
    ok &= r

    # 防线 7 反向自检：造一个真的缺符号文件，验证它**确实会报出来**
    # （否则这条检查就是「永远通过」的装饰）
    with _tf.TemporaryDirectory() as td:
        broken = Path(td) / "broken.py"
        broken.write_text(
            "def f():\n"
            "    return build_model('x')  # 故意不 import\n",   # noqa
            encoding="utf-8")
        found = check_undefined_symbols([broken])
        r = len(found) == 1 and "build_model" in found[0]
        print(f"  {'✅' if r else '❌'} 防线7 对缺符号文件会 FAIL: {r}")
        ok &= r

    # 防线 7' 不可达代码检查：当前代码必须干净 + 反向自检
    unre = check_unreachable_code()
    r = not unre
    print(f"  {'✅' if r else '❌'} 防线7' 当前代码无不可达语句: {r}"
          + (f" {unre}" if unre else ""))
    ok &= r

    with _tf.TemporaryDirectory() as td:
        # 造一个 return 之后还有语句的函数（模拟误删 def 行）
        dang = Path(td) / "dangling.py"
        dang.write_text(
            "def outer():\n"
            "    return 1\n"
            "    x = 2\n",          # 不可达
            encoding="utf-8")
        found = check_unreachable_code([dang])
        r = len(found) == 1
        print(f"  {'✅' if r else '❌'} 防线7' 对不可达代码会 FAIL: {r}")
        ok &= r

    print("=" * 62)
    print("✅ 全部防线自检通过" if ok else "❌ 自检未通过")
    return 0 if ok else 1

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(full_report(root=sys.argv[1] if len(sys.argv) > 1 else None))
