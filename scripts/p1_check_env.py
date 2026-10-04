"""P1 环境自检（在 4060 目标机上先跑这个，再花 GPU 时间）。

检查项：
  1. Python / PyTorch / CUDA 是否可用、显卡名、显存
  2. timm 是否可用，能否列出候选骨干
  3. 数据集是否已就位（读 data/p1_dataset_manifest.json；没有就按默认路径找）
  4. 小显存冒烟：实跑一次"合成数据 → 前向 → 反向"，确认 GPU 真能训（不是装了 CPU 版 torch）

用法：
    python scripts/p1_check_env.py
    python scripts/p1_check_env.py --data-root D:/datasets/fdm-defect/data

退出码 0 = 全绿；非 0 = 有 FAIL（会指出先修哪一项）。
"""
import argparse
import json
import sys
from pathlib import Path

OK, FAIL = "  ✅", "  ❌"
results = []


def check(name, fn):
    try:
        status, msg = fn()
    except Exception as e:
        status, msg = False, f"{type(e).__name__}: {e}"
    results.append((status, name, msg))
    print(f"{OK if status else FAIL} {name}: {msg}")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="数据集根目录（含类别子目录）")
    ap.add_argument("--batch", type=int, default=32, help="显存冒烟测试的 batch size")
    ap.add_argument("--img", type=int, default=224, help="显存冒烟测试的输入尺寸")
    args = ap.parse_args()

    print("=" * 60)
    print("P1 环境自检 · 3D 打印缺陷分类")
    print("=" * 60)

    # 1) Python
    check("Python 版本", lambda: (sys.version_info[:2] >= (3, 9),
                                  f"{sys.version.split()[0]}"))

    # 2) torch + CUDA
    def torch_check():
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "N/A"
        mem = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB" if cuda else "-"
        return cuda, (f"torch={torch.__version__} cuda={torch.version.cuda} "
                      f"device={name} mem={mem}")
    has_cuda = check("PyTorch + CUDA", torch_check)
    if not has_cuda:
        print("\n❌ CUDA 不可用 —— 先修这个再往下。常见原因：装成了 CPU 版 torch。")
        print("   ⚠️ 本机（Python 3.13）请用 cu126，不要用 cu124 —— cu124 索引下cp313")
        print("      只有 torch 2.6/tv 0.21（老版本）；cu126 有 torch 2.9.1/tv 0.29.1。")
        print("   重装：pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu126")

    # 3) timm
    def timm_check():
        import timm
        n = len(timm.list_models())
        return True, f"timm={timm.__version__} 可用模型 {n} 个"
    check("timm 骨干库", timm_check)

    # 4) 数据
    def data_check():
        root = args.data_root
        if root is None:
            mf = Path("data/p1_dataset_manifest.json")
            if mf.exists():
                root = json.load(open(mf, encoding="utf-8")).get("root")
            else:
                root = "D:/datasets/fdm-defect"
        root = Path(root)
        if not root.exists():
            return False, f"未找到 {root}（先跑 scripts/p1_fetch_dataset.py）"
        # ⚠️ 2026-10-04 修正：Kaggle zip 解压后真实结构是
        #    <root>/FDM-3D-Printing-Defect-Dataset/data/<类别>/*.jpg（**两层**嵌套），
        #    原先只 iterdir 一层 → 永远数到 0 张 → 假 FAIL。
        #    复用 guards.probe_dataset 的层级下探（它本就是为这个场景写的）。
        # ⚠️ 用本地文件加载而非 `from src.vision...`：脚本在 scripts/ 下直接运行时，
        #    sys.path[0] 是 scripts/，包导入会失败——之前用 `except: pass` 吞掉了，
        #    结果下探静默失效、依然报 0 张。**别再吞异常。**
        import importlib.util
        _g = Path(__file__).resolve().parent.parent / "src" / "vision" / "guards.py"
        _spec = importlib.util.spec_from_file_location("p1_guards", _g)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["p1_guards"] = _mod          # dataclass 解析需要先注册
        _spec.loader.exec_module(_mod)
        probe = _mod.probe_dataset(root, strict=False)
        if probe.root != root:
            root = probe.root
        counts = {}
        for d in root.iterdir():
            if d.is_dir():
                n = len([f for f in d.iterdir()
                         if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
                if n:
                    counts[d.name] = n
        total = sum(counts.values())
        exp = {"Warping": 538, "Cracking": 472, "Stringing": 447,
               "Layer_shifting": 364, "Off_platform": 91}
        # ⚠️ 2026-10-04 修正：原先用 counts == exp 严格相等，一个无关文件就会误报 FAIL。
        # 改为「每类达标且总量达标即 PASS」，精确计数另行打印供人工核对。
        missing = [c for c in exp if c not in counts]
        short = {c: (counts.get(c, 0), n) for c, n in exp.items()
                 if counts.get(c, 0) < n}
        ok = not missing and not short and total >= sum(exp.values())
        detail = f"{root} 共 {total} 张 {counts}"
        if ok:
            exact = "（与基准完全一致）" if counts == exp else \
                    "（≥ 基准，含额外文件，不影响使用）"
            return True, detail + "  " + exact
        msg = detail + "  ⚠️ 数据未就位或不完整"
        if missing:
            msg += f"；缺类别目录: {missing}"
        if short:
            msg += f"；张数不足: {short}"
        return False, msg
    check("数据集", data_check)

    # 5) GPU 实跑冒烟（合成数据，几秒）
    def smoke():
        import torch
        import timm
        if not torch.cuda.is_available():
            return False, "跳过（无 CUDA）"
        model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5)
        model.cuda().train()
        x = torch.randn(args.batch, 3, args.img, args.img).cuda()
        y = torch.randint(0, 5, (args.batch,)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.empty_cache()
        return True, (f"bs={args.batch} @ {args.img}px 前向+反向通过；"
                      f"峰值显存 {peak:.2f} GB（4060 8GB 建议峰值 < 6 GB）")
    check("GPU 训练冒烟", smoke)

    print("=" * 60)
    n_fail = sum(1 for s, _, _ in results if not s)
    if n_fail == 0:
        print("✅ 全绿：可以开始写 P1 训练脚本")
        print("   下一步：先加载 kaggle-vision-pipeline-guards skill（防静默失败），再写训练脚本")
        sys.exit(0)
    print(f"❌ 有 {n_fail} 项未通过：按上面提示修完再开训（尤其是 CUDA 与数据集）")
    sys.exit(1)


if __name__ == "__main__":
    main()
