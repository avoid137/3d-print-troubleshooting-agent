"""P1 数据集下载与校验：Kaggle DC-02R（wengmhu / FDM 3D Printing Defect Dataset, MIT）。

设计目的：换机器后**一条命令**把 P1 主数据源拉下来，并用实测类别计数做完整性校验，
避免"以为下载好了其实缺类/缺图"导致的静默失败。

实测基准（2026-09-28 经 Kaggle API 文件清单核实，必须与下载结果一致）：
    Warping 538 / Cracking 472 / Stringing 447 / Layer_shifting 364 / Off_platform 91
    合计 1,912 张；包体约 6.3 GB

用法：
    python scripts/p1_fetch_dataset.py                       # 默认下载到 D:/datasets/fdm-defect
    python scripts/p1_fetch_dataset.py --dir D:/mydata       # 自定义目录
    python scripts/p1_fetch_dataset.py --check-only --dir X  # 只校验已下载目录

凭据（三选一，脚本会按序探测，找不到就明确报错并给出指引）：
    1) 环境变量 KAGGLE_USERNAME + KAGGLE_KEY（经典 kaggle.json 方式）
    2) ~/.kaggle/kaggle.json
    3) 新版 access_token：环境变量 KAGGLE_ACCESS_TOKEN，或 ~/.kaggle/access_token 文件
       ⚠️ access_token 是密钥，**不要**写进仓库、不要贴进对话。
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATASET_SLUG = "wengmhu/fdm-3d-printing-defect-dataset"

# 实测基准（dataset_cards.md DC-02R；Kaggle 上游 README 写 Cracking=427 是笔误，实为 472）
EXPECTED = {
    "Warping": 538,
    "Cracking": 472,
    "Stringing": 447,
    "Layer_shifting": 364,
    "Off_platform": 91,
}
EXPECTED_TOTAL = 1912

# ⚠️ 2026-10-04 实测坑：Kaggle API 元数据里的 `totalBytes` **不可作为下载完整性基准**。
#   元数据报 6,293,272,351 B，但 Google Storage 实际对象
#   （`x-goog-stored-content-length`）是 6,279,686,097 B，差 13,586,254 B。
#   后果：8 线程分片下载后按 totalBytes 比对会误报 SIZE_MATCH=NO，
#   白白补下 13 MB（且 Range 请求直接被服务器拒为 InvalidRange）。
#   **正确基准 = zip 条目数 1,912 + 类内张数 + zipfile.testzip() CRC 自检。**
#   详见 exp/p1_vision_result.md §1.0。
ZIP_EXPECTED_ENTRIES = 1912


def human(n):
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.1f} MB"


def free_space(path):
    total, used, free = shutil.disk_usage(path)
    return free


def find_data_root(base: Path):
    """Kaggle 解压后通常多一层目录（.../data/<类别>/*.jpg）。逐层下探找含类别子目录的那层。"""
    best, best_hit = None, 0
    for p in [base] + [d for d in base.rglob("*") if d.is_dir()]:
        hit = sum(1 for c in EXPECTED if (p / c).is_dir())
        if hit > best_hit:
            best, best_hit = p, hit
    return best if best_hit >= 3 else None


def verify(root: Path):
    print(f"\n[校验] 根目录: {root}")
    ok = True
    total = 0
    detail = {}
    for cls, exp in EXPECTED.items():
        d = root / cls
        if not d.is_dir():
            print(f"  ❌ 缺类别目录: {cls}")
            ok = False
            detail[cls] = None
            continue
        imgs = [f for f in d.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        n = len(imgs)
        total += n
        detail[cls] = n
        flag = "✅" if n == exp else "❌"
        if n != exp:
            ok = False
        print(f"  {flag} {cls:16s} 实测 {n:5d}  期望 {exp:5d}")
    print(f"  {'✅' if total == EXPECTED_TOTAL else '❌'} 合计 实测 {total}  期望 {EXPECTED_TOTAL}")
    if total != EXPECTED_TOTAL:
        ok = False

    # 附加体检：图像可读性 + 尺寸抽样（不解码全部，只抽样 30 张防损坏文件）
    try:
        from PIL import Image
        import random
        sample = []
        for cls in EXPECTED:
            d = root / cls
            if not d.is_dir():
                continue
            fs = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            random.seed(0)
            sample += random.sample(fs, min(6, len(fs)))
        bad = []
        sizes = set()
        for f in sample:
            try:
                with Image.open(f) as im:
                    im.verify()
                    sizes.add(im.size)
            except Exception as e:
                bad.append((f.name, type(e).__name__))
        print(f"  [抽样] {len(sample)} 张：损坏 {len(bad)} 张；出现尺寸 {sorted(list(sizes))[:5]}")
        if bad:
            print(f"    ⚠️ 损坏样例: {bad[:3]}")
    except ImportError:
        print("  [抽样] 跳过（未安装 Pillow）")

    if ok:
        print("\n✅ 数据集校验通过：类别与张数与 DC-02R 实测基准完全一致")
    else:
        print("\n❌ 数据集校验未通过：不要开训，先修数据")
    return ok, detail


def download(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[下载] 目标目录: {dest}")
    print(f"[检查] 目标盘剩余空间: {human(free_space(dest))}（包体约 6.3 GB，需预留 ≥ 10 GB）")
    if free_space(dest) < 10 * 1024**3:
        print("❌ 剩余空间不足 10 GB，换盘或清理后再下。")
        sys.exit(2)

    # ---- 凭据探测（不打印密钥内容）----
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    has_token_file = (Path.home() / ".kaggle" / "access_token").exists()
    env_user = bool(os.getenv("KAGGLE_USERNAME"))
    env_key = bool(os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_ACCESS_TOKEN"))
    print(f"[凭据] kaggle.json={has_json}  access_token文件={has_token_file}  "
          f"环境变量={'有' if (env_user and env_key) else '无'}")
    if not (has_json or has_token_file or (env_user and env_key)):
        print(
            "\n❌ 未找到 Kaggle 凭据。三选一配置：\n"
            "  1) 经典：~/.kaggle/kaggle.json = {\"username\":\"<你的>\",\"key\":\"<API key>\"}\n"
            "  2) 新版 token：环境变量 KAGGLE_ACCESS_TOKEN + KAGGLE_USERNAME\n"
            "  3) 从旧机器拷贝 ~/.kaggle/access_token 到同路径（密钥不要进仓库）\n"
        )
        sys.exit(3)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ 未安装 kaggle：pip install kaggle")
        sys.exit(4)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"❌ Kaggle 认证失败: {type(e).__name__}: {e}")
        sys.exit(5)

    print("[下载] 开始拉取（6.3 GB，视网络 10~40 分钟）...")
    try:
        api.dataset_download_files(DATASET_SLUG, path=str(dest), unzip=True, quiet=False)
    except Exception as e:
        print(f"❌ 下载失败: {type(e).__name__}: {e}")
        sys.exit(6)
    print("[下载] 完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/datasets/fdm-defect",
                    help="数据集落盘目录（默认 D:/datasets/fdm-defect，别放 C 盘）")
    ap.add_argument("--check-only", action="store_true", help="只校验，不下载")
    ap.add_argument("--no-verify", action="store_true", help="下载后跳过校验")
    args = ap.parse_args()

    dest = Path(args.dir)
    if not args.check_only:
        download(dest)

    root = find_data_root(dest)
    if root is None:
        print(f"\n❌ 在 {dest} 下未找到含类别子目录的数据根（期望 data/<类别>/*.jpg）")
        print(f"   目录实际内容: {[p.name for p in dest.iterdir()][:10]}")
        sys.exit(7)

    if args.no_verify:
        print(f"\n[跳过校验] 数据根: {root}")
        return
    ok, detail = verify(root)
    out = {"root": str(root), "ok": ok, "detail": detail, "expected": EXPECTED}
    Path("data/p1_dataset_manifest.json").parent.mkdir(exist_ok=True, parents=True)
    Path("data/p1_dataset_manifest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[产出] data/p1_dataset_manifest.json（数据根与实测计数，供训练脚本读取）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
