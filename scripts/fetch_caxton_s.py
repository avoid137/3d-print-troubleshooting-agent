"""下载 Caxton-S（Kaggle adritganeriwala/caxton-s-semantic-3d-printing-defect-dataset）。

为什么单独写这个脚本，而不复用 p1_fetch_dataset.py
----------------------------------------------------
`p1_fetch_dataset.py` 走的是 `kaggle` 包的 `dataset_download_files()`，那条路**需要凭据**
（本机 `~/.kaggle/` 不存在、三个环境变量全无 → `sys.exit(3)`）。
但 2026-10-04 实测并已写入记忆：**Kaggle 公开数据集的整包下载端点匿名可下**
（`GET /api/v1/datasets/download/<owner>/<slug>` 会302 到 GCS 签名 URL）。

🔴 本脚本实测到的三个坑（都写在这里，别重走）：
① `curl -I`（HEAD）打同一端点返回 **404** → 端点只支持 GET。
   「404 → 判定下不了」是错的，会白放弃一次。
② 必须用 **Range GET**（`-r 0-1023`）先探活；直接整包GET 才知道能不能下。
③ 走 GCS 签名 URL，**签名有效期 259200 s（3 天）**，中途断了重跑要重新取签名。

完整性判据（沿用项目铁律：**Kaggle `totalBytes` 不可作基准**）
--------------------------------------------------------------
`totalBytesNullable = 8,011,887,621`，但 GCS 实际对象大小与之有差（DC-02R 上实测差 13 MB）。
所以判据 = ① zip 能打开 ② **CRC 自检** `testzip()` ③ **各类张数**对上 README 基准。

用法
------------------------------------------------------------------
    python scripts/fetch_caxton_s.py --dir D:/datasets/caxton-s
    python scripts/fetch_caxton_s.py --dir D:/datasets/caxton-s --threads 8
    python scripts/fetch_caxton_s.py --dir D:/datasets/caxton-s --check-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG = "adritganeriwala/caxton-s-semantic-3d-printing-defect-dataset"
API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
# 换存根/大小写
PROBE_URL = f"https://www.kaggle.com/api/v1/datasets/view/{SLUG}"

# README 声明的基准（Caxton-S v1，2026-04-03）
README_COUNTS = {
    "good_print": 18175,
    "under_extrusion": 66084,
    "over_extrusion": 84943,
    "spaghetti": 6420,
    "warping": 9790,
    "stringing": 2358,
}
README_TOTAL = 187770  # 注意：逐类相加 = 187,770

#: 实测目录名 → README 类别名（下载后核对用；不一致会显式报出，不静默）
DIR2CLASS = {
    "good_print": "Good Print (正常件)",
    "under_extrusion": "Under-Extrusion",
    "over_extrusion": "Over-Extrusion",
    "spaghetti": "Spaghetti",
    "warping": "Warping",
    "stringing": "Stringing",
}


def human(n: float) -> str:
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.1f} MB"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def curl(args: list, timeout: int = 60) -> subprocess.CompletedProcess:
    # 🔴 2026-10-06 修**自己写出来的静默 bug**：`probe()` 里调用时漏传 URL，
    #   `curl -o file -w ...` 没有 URL → rc=2 + stdout 空。
    #   现象是「探活失败」，看起来像"端点不支持/网络被墙"，
    #   实际是**参数漏传**。这类错误不报错、只返回空 → 极易误判。
    #   对策：辅助函数入口就断言 URL 存在，让它当场抛。
    if not any(a.startswith("http") for a in args):
        raise ValueError(
            f"curl() 调用缺 URL，args={args!r}。"
            "（漏传 URL 时 curl rc=2 且 stdout 为空，"
            "会被误读成网络/权限问题——本项目已踩过一次）")
    return subprocess.run(
        ["curl", "-s", "-L", "--max-time", str(timeout)] + args,
        capture_output=True,
    )


def probe(dest: Path) -> tuple:
    """探活：拿元数据 + 用 Range GET 试首字节。

    🔴 不用 HEAD（实测 404）。元数据 totalBytes 只作参考，不作完整性判据。
    """
    print("[探活] 元数据 …")
    r = curl([PROBE_URL], timeout=60)
    if r.returncode != 0 or not r.stdout:
        print(f"❌ 元数据获取失败 rc={r.returncode}")
        return None, 0
    try:
        meta = json.loads(r.stdout)
    except json.JSONDecodeError:
        print("❌ 元数据非JSON（可能被拦截或需要凭据）")
        return None, 0
    tb = int(meta.get("totalBytes") or 0)
    print(f"  title   : {meta.get('title')}")
    print(f"  license : {meta.get('licenseName')}")
    print(f"  version : {meta.get('currentVersionNumber')}"
          f"  updated: {meta.get('lastUpdated')}")
    print(f"  totalBytes(参考,不作判据): {human(tb)}")

    print("[探活] Range GET 首字节 …")
    out = dest / "_probe.bin"
    dest.mkdir(parents=True, exist_ok=True)
    # 🔴 2026-10-06 实测：同一命令**第一次**探活返回空（rc≠0 / 无 body），
    #   隔几秒重跑同一个 curl 就正常 206。Kaggle CDN 侧偶发空响应，
    #   **不重试会把"瞬时失败"误判成"端点不支持"**（本项目铁律：
    #   静默算错比报错危险）。故探活一律重试 3 次。
    ok = False
    for attempt in range(1, 4):
        r = curl(["-r", "0-1023", "-o", str(out),
                  "-w", "%{http_code} %{size_download}", API_URL], timeout=60)
        info = (r.stdout or b"").decode(errors="ignore").strip()
        got = out.stat().st_size if out.exists() else 0
        print(f"  试{attempt}: rc={r.returncode} http/size={info!r} 落地={got}B"
              + (f" err={(r.stderr or b'').decode(errors='ignore')[:120]!r}"
                 if r.returncode != 0 else ""))
        if got == 1024:
            with out.open("rb") as f:
                magic = f.read(2)
            if magic == b"PK":
                ok = True
                print(f"  ✅ 206 + 魔数 PK → zip 可下（真实大小见下）")
                break
            print(f"  ❌ 魔数 {magic!r} 不是 zip")
        if attempt < 3:
            time.sleep(4)
    out.unlink(missing_ok=True)
    if not ok:
        print("  ❌ 3 次探活均失败")
        return None, tb
    return meta, tb


def real_size(dest: Path, meta_bytes: int) -> int:
    """从 GCS 签名 URL 取真实 Content-Length。

    先发一个 1 字节 Range 请求，读响应头 `Content-Range: bytes 0-0/TOTAL`。
    （元数据 totalBytes 与真实值有差，DC-02R 上实测差 13 MB。）

    🔴 重试 3 次：探活阶段实测过 Kaggle CDN 偶发空响应。
       若拿不到就**回退到 meta_bytes 并显式告警**——不能静默用不可信值，
       否则分片大小会算错，合并后必然对不上。
    """
    for attempt in range(1, 4):
        r = curl(["-r", "0-0", "-D", "-", "-o", os.devnull, API_URL],
                 timeout=60)
        hdr = (r.stdout or b"").decode(errors="ignore")
        for line in hdr.splitlines():
            if line.lower().startswith("content-range:"):
                return int(line.split("/")[-1].strip())
        print(f"  [real_size] 试{attempt}未取到 Content-Range，重试…")
        time.sleep(4)
    print(f"  ⚠️  real_size 取不到，回退用元数据 totalBytes={meta_bytes}。"
          f"（该值已知与真实值有差，分片合并后可能对不上——届时会显式报错，"
          f"不会静默当成品）")
    return meta_bytes


def download(dest: Path, threads: int, resume: bool) -> Path:
    zip_path = dest / "caxton_s.zip"
    meta, tb = probe(dest)
    if meta is None:
        sys.exit(2)

    total = real_size(dest, tb)
    print(f"\n[实测] GCS 真实大小 {human(total)}（元数据 {human(tb)}，"
          f"差 {total - tb:+d} B → 印证 totalBytes 不可作判据）")

    need_gb = total / 1024**3 * 1.05 + 1
    print(f"[空间] 目标盘剩余 {free_gb(dest):.1f} GB / 需要约 {need_gb:.1f} GB")
    if free_gb(dest) < need_gb:
        print("❌ 空间不足，换盘或清理后重跑。")
        sys.exit(3)

    if zip_path.exists() and resume and zip_path.stat().st_size == total:
        print(f"[续传] 已存在且大小一致，跳过下载：{zip_path}")
        return zip_path
    if zip_path.exists() and not resume:
        print(f"[续传] 关闭续传，重下（覆盖 {zip_path.stat().st_size} B 的残片）")
        zip_path.unlink()

    # ---- 分片并发：Kaggle CDN 支持 Range；实测 8 并发最优，16 无增益 ----
    #🔴 2026-10-06 实测两个必须处理的坑（首次跑挂了 1.5 h才发现）：
    #   ① **必须设 `--speed-limit/--speed-time`**：不带超时的 curl 会在
    #      速率掉到0 时**永久挂起**（实测 7 个进程全停、25 s 零字节，
    #      文件时间戳停在 14:42，进程却一直"running"）。
    #      → `--speed-limit 10240 --speed-time 60` = 低于 10 KB/s 持续 60 s 就自杀重试。
    #      ⚠️ 这与「限流时 rate=0 会ZeroDivisionError」是同一类问题：
    #      **0 速率必须被当成异常处理，不能当成"还在慢慢下"**。
    #   ② **必须支持断点续传**：此前 `curl -o` 覆盖 = 从零重下，
    #      挂了 1.5 h 才发现 → 2.4 GB 白费。curl 加 `-C -` 即可续传。
    chunk = (total + threads - 1) // threads
    parts = []
    print(f"[下载] {threads} 线程分片，每片 {human(chunk)} …")
    procs = []
    for i in range(threads):
        lo = i * chunk
        hi = min(total - 1, lo + chunk - 1)
        if lo > hi:
            break
        p = dest / f"_part{i:02d}"
        parts.append((p, lo, hi))
        # 🔴 2026-10-06 连续两次栽在同一种 bug 上：`parts` 是 (path, lo, hi)，
        #   进度循环里却按 `(p, lo, hi)` 之外的顺序解包 →拿到 int当路径用。
        #   报错虽响，但已浪费两轮 8线程下载。**在源头断言形状。**
        assert isinstance(p, Path), f"parts[0] 必须是 Path，实为 {type(p)}"

        # ---- 断点续传：**手工算区间**，不用 curl -C - ----
        # 🔴 实测坑：`curl -C -` 与 `-r` **不能共用**，curl 直接报
        #   `option -r: is badly used here` 且**静默不续传**（文件大小不变）。
        #   若照抄会以为"续传了"实则从头覆盖，白等几小时。
        #   正解：分片已有 N 字节 → 只需拉 [lo+N, hi]，追加写。
        want_here = hi - lo + 1
        already = p.stat().st_size if p.exists() else 0
        if already > want_here:
            # 超长 = 上次换过范围或文件损坏 → 删掉重下，绝不"截断"凑数
            print(f"  [续传] _part{i:02d} 现有 {already} B > 应有 {want_here} B"
                  f" → 判定损坏，删除重下")
            p.unlink()
            already = 0
        if already == want_here:
            print(f"  [续传] _part{i:02d} 已完整 {human(already)}，跳过")
            procs.append((i, lo, hi, p, None))
            continue
        if already:
            print(f"  [续传] _part{i:02d} 已有 {human(already)} / "
                  f"{human(want_here)}，只补拉缺口")
        start = lo + already
        cmd = [
            "curl", "-s", "-L",
            "--retry", "20", "--retry-delay", "5",
            "--retry-all-errors",
            # 🔴 低速自杀：低于 10 KB/s 持续 60 s 放弃本次连接（触发 retry）。
            #   不带这两个参数时速率掉到 0 会**永久挂起**（实测 1.5 h 零字节、
            #   进程仍报 running）——0 速率必须当异常处理。
            "--speed-limit", "10240", "--speed-time", "60",
            "-r", f"{start}-{hi}",
            # 追加写（`-o -` 追加不行，curl 会覆盖）→ 先下到临时文件再合并
            "-o", str(p) + ".seg", API_URL,
        ]
        procs.append((i, lo, hi, p,
                      subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)))

    t0 = time.time()
    done_bytes = 0
    # parts 的元素是 (path, lo, hi) → **path 是第 0 位**，不是第 2 位。
    # 🔴 本次已栽两次：写成 `for _, _, p in parts` 会把 `hi`(int) 当路径用，
    #   报 AttributeError。解包顺序错是本项目最常见的静默/报错混合型bug。
    last_size = -1
    last_change = time.time()
    STALL_LIMIT = 300  # 秒：5 分钟零增长视为卡死
    #🔴 procs 里可能有 `pr=None`（分片已完整被跳过），所有遍历必须过滤掉
    alive = [t for t in procs if t[4] is not None]
    while any(pr.poll() is None for _, _, _, _, pr in alive):
        time.sleep(10)
        done_bytes = sum(
            (p.stat().st_size if p.exists() else 0)
            + (Path(str(p) + ".seg").stat().st_size
               if Path(str(p) + ".seg").exists() else 0)
            for p, _, _ in parts)
        rate = done_bytes / max(time.time() - t0, 1e-9)
        left = (total - done_bytes) / rate if rate > 0 else float("inf")
        # 🔴 rate=0 时不算 eta（否则 ZeroDivisionError，且把"正在被限流"的现场一起带走）
        eta = f"ETA {left/60:.1f} min" if rate > 0 else "ETA 未知（速率 0，可能被限流）"
        print(f"  {done_bytes/1e9:6.2f}/{total/1e9:.2f} GB "
              f"({done_bytes/total*100:5.1f}%) {human(rate)}/s  {eta}")

        # 🔴 卡死检测：进程还在跑但字节数不动 → 与其干等 1.5 h，不如立刻报错
        if done_bytes != last_size:
            last_size, last_change = done_bytes, time.time()
        elif time.time() - last_change > STALL_LIMIT:
            print(f"\n🔴 **卡死：{STALL_LIMIT}s 内字节数零增长**"
                  f"（停在 {done_bytes/1e9:.2f} GB）"
                  f"→ curl 连接已挂死（低速但未触发 --speed-limit，或CDN 半开）")
            print("   已保留所有分片，重跑本脚本会**断点续传**，不会从零开始。")
            for _, _, _, _, pr in alive:
                pr.kill()
            sys.exit(7)

        if time.time() - t0 > 60 * 60 * 12:
            print("❌ 超时 12 小时仍未完成，中止（分片已保留，可续传）。")
            for _, _, _, _, pr in alive:
                pr.kill()
            sys.exit(4)

    # ---- 各分片收尾：把 .seg 追加回分片本体 ----
    # 🔴 顺序必须按索引 i 升序，且**每一片都独立追加**——
    #   缺这一步会得到"分片是完整的但少了中间一段"的zip，
    #   而 zipfile 只在读中央目录时才报错，CRC 自检能过一部分，
    #   表现为"部分文件能读、部分读不了"，极难定位。
    for i, lo, hi, p, pr in procs:
        seg = Path(str(p) + ".seg")
        if pr is None or not seg.exists():
            seg.unlink(missing_ok=True)
            continue
        if seg.stat().st_size:
            with p.open("ab") as dst, seg.open("rb") as src:
                shutil.copyfileobj(src, dst, 1 << 22)
        seg.unlink(missing_ok=True)

    bad = []
    for i, lo, hi, p, pr in alive:
        want = hi - lo + 1
        got = p.stat().st_size if p.exists() else 0
        # 🔴 续传后 curl 可能已正常退出但返回非 0（"range not satisfiable"
        #   这类在分片已下完时无害）。**判据是最终字节数，不是返回码。**
        if got != want:
            bad.append((i, f"size {got} != {want}", pr.returncode))
    if bad:
        print(f"❌ 分片未达预期大小: {bad}")
        print("   → 重跑本脚本会**断点续传**（curl -C -），已下的部分不会重下。")
        sys.exit(5)

    print("[合并] 拼接分片 …")
    with zip_path.open("wb") as out:
        for p, _, _ in parts:
            with p.open("rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    for p, _, _ in parts:
        p.unlink(missing_ok=True)

    got = zip_path.stat().st_size
    if got != total:
        print(f"❌ 合并后大小 {got} != {total}")
        sys.exit(6)
    print(f"[完成] {zip_path} {human(got)}")
    return zip_path


def verify(zip_path: Path) -> tuple:
    """zip CRC + 各类张数校验。**没有基准的类别显式报缺，不补数字。**"""
    print(f"\n[校验] 打开 zip 并做 CRC 自检（8 GB，需几分钟）…")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            print(f"❌ CRC 自检失败，首个坏条目：{bad}")
            return False, {}
        names = [n for n in z.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  CRC 全过，用时 {time.time()-t0:.0f}s")

    counts: dict = {}
    for n in names:
        parts = Path(n).parts
        # 形如 Kaggle_3D_Print_Defect_Dataset/<类别>/<文件>.jpg
        for p in parts:
            if p in README_COUNTS:
                counts[p] = counts.get(p, 0) + 1
                break
    print(f"\n[校验] 图像条目共 {len(names)} 张（README 声明 {README_TOTAL}）")
    ok = True
    detail = {}
    for d, cls in DIR2CLASS.items():
        n = counts.get(d, 0)
        exp = README_COUNTS[d]
        flag = "✅" if n == exp else "❌"
        if n != exp:
            ok = False
        detail[d] = {"count": n, "expected": exp, "match": n == exp}
        print(f"  {flag} {d:18s} 实测 {n:6d}  期望 {exp:6d}  ({cls})")
    unseen = set(counts) - set(README_COUNTS)
    if unseen:
        print(f"  ⚠️ 出现 README 未声明的目录: {sorted(unseen)}（台账需同步，勿静默）")
    if len(names) != README_TOTAL:
        ok = False
        print(f"  ❌ 图像总数 {len(names)} != README {README_TOTAL}")
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/datasets/caxton-s")
    ap.add_argument("--threads", type=int, default=8,
                    help="分片并发数（实测 8 最优，16 无增益）")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="只校验已下载的 zip，不重下")
    a = ap.parse_args()

    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "caxton_s.zip"

    print("=" * 68)
    print("Caxton-S 下载与校验（Kaggle 公开数据集，匿名可下）")
    print("=" * 68)
    print(f"slug  : {SLUG}")
    print(f"许可  : CC BY 4.0 —— 🔴 使用需署名 Brion & Pattinson (2022), "
          f"DOI 10.17863/CAM.84082")
    print(f"落盘  : {dest}\n")

    if a.check_only:
        if not zip_path.exists():
            print(f"❌ 不存在 {zip_path}")
            return 1
    else:
        zip_path = download(dest, a.threads, resume=not a.no_resume)

    ok, detail = verify(zip_path)
    out = {
        "slug": SLUG,
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "license": "CC BY 4.0 (需署名)",
        "attribution": ("Brion, D., & Pattinson, S. (2022). Data set for "
                        "'Generalisable 3D printing error detection and "
                        "correction via multi-head neural networks'. "
                        "University of Cambridge Apollo Repository. "
                        "https://doi.org/10.17863/CAM.84082"),
        "ok": ok,
        "per_class": detail,
        "classes": DIR2CLASS,
    }
    mf = ROOT / "data" / "caxton_s_manifest.json"
    mf.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产出] {mf}")
    print("✅ 校验通过" if ok else "❌ 校验未通过，不要开训，先查数据")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
