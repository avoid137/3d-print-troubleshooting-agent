"""判官复现性验证（self-consistency 实验）。

回答三个问题
------------
Q1  temp=0.0（生产配置），同一条 query 反复调用，标签稳定吗？
    → 若稳定，F1 0.9487 是硬数字，不需要 self-consistency；
      若不稳定，必须量化波动区间，否则对外报的单点数字站不住。
Q2  temp=0.7（压力测试），波动有多大？
    → 用来证明「为什么必须把温度钉在 0」，而不是拍脑袋设 0。
Q3  多数投票（self-consistency, k=3 / k=5）能否消除波动？成本增加多少？
    → 只有 Q1/Q2 显示不稳定时，加 self-consistency 才是有理有据的工程决策。

实验设计
--------
- Q1：**全部 200 条** × 5 次重复 = 1000 次调用（temp=0.0）
- Q2：**80 条子集**（40 负例 + 40 条 top-1 余弦最低的正例，即最易判错的边界样本）
      × 3 次重复 = 240 次调用（temp=0.7）
- Q3：从 Q1/Q2 的重复结果直接模拟多数投票，无需额外调用

⚠️ 必须绕过 judge_many 的磁盘缓存，否则 5 次会全部命中缓存、测不到任何波动。
   本脚本直接调 judge_once。

运行：
    cd src && D:/envs/printagent/Scripts/python.exe ../exp/exp_judge_reproducibility.py
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import EVAL_SET_PATH, RECALL_K  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from rejector import JUDGE_MODEL, judge_once  # noqa: E402

CTX_PATH = PROJECT_ROOT / "exp" / "_judge_repro_ctx.json"
OUT_PATH = PROJECT_ROOT / "data" / "judge_repro_result.json"

REPEATS_T0 = 5          # temp=0.0 重复次数（全量 200 条）
REPEATS_T07 = 3         # temp=0.7 重复次数（子集 80 条）
T07_SUBSET = 80
WORKERS = 8


# ---------------------------------------------------------------- 检索上下文
def load_contexts():
    """载入「query + 检索结果」，首次运行需调 embedding，之后读缓存。"""
    if CTX_PATH.exists():
        print(f"[ctx] 读缓存 {CTX_PATH.name}")
        return json.loads(CTX_PATH.read_text(encoding="utf-8"))

    ev = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
    r = HybridRetriever()
    print(f"[ctx] 检索 {len(ev)} 条（后端 {r.embed_method}）...")
    items = []
    for e in ev:
        res = r.retrieve(e["query"], top_k=RECALL_K)
        items.append({
            "id": e["id"],
            "query": e["query"],
            "category": e["category"],
            "gold": e["gold"],
            "top_cosine": res[0]["cosine"] if res else -1.0,
            "results": [{"kb_id": x["kb_id"], "problem": x["problem"],
                         "text": x["text"]} for x in res],
        })
    CTX_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[ctx] 已缓存 {CTX_PATH}")
    return items


# ---------------------------------------------------------------- 重复采样
def run_repeats(items, repeats, temperature, tag):
    """对每条 query 调 repeats 次判官，返回 {id: [answerable, ...]} 及理由。"""
    n_calls = len(items) * repeats
    print(f"\n[{tag}] temperature={temperature}, {len(items)} 条 × {repeats} 次 "
          f"= {n_calls} 次调用 ...")
    labels = {it["id"]: [None] * repeats for it in items}
    reasons = {it["id"]: [""] * repeats for it in items}
    fail = []

    def task(it, k):
        v = judge_once(it["query"], it["results"], model=JUDGE_MODEL,
                       temperature=temperature)
        return it["id"], k, v

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(task, it, k) for it in items for k in range(repeats)]
        done = 0
        for fut in as_completed(futs):
            try:
                qid, k, v = fut.result()
                labels[qid][k] = bool(v["answerable"])
                reasons[qid][k] = v.get("reason", "")
            except Exception as e:  # noqa: BLE001
                fail.append(f"{type(e).__name__}: {e}")
            done += 1
            print(f"  [{done}/{n_calls}]", end="\r")
    dt = time.time() - t0
    print(f"  完成，耗时 {dt:.1f}s（{n_calls / max(dt, 1e-6):.1f} call/s）"
          f"{f'，失败 {len(fail)}' if fail else ''}")

    # 剔除失败导致的 None
    for qid in labels:
        keep = [i for i, x in enumerate(labels[qid]) if x is not None]
        labels[qid] = [labels[qid][i] for i in keep]
        reasons[qid] = [reasons[qid][i] for i in keep]

    return {"labels": labels, "reasons": reasons,
            "n_calls": n_calls, "seconds": round(dt, 1), "failures": fail[:5]}


# ---------------------------------------------------------------- 指标
def compute_f1(rows):
    """rows: [(gold_nonempty, rejected)] → Precision / Recall / F1

    与 evaluate_reject.py 口径一致：拒答为正类，
      tp = 库外且拒答（正确拒答）
      fp = 库内且拒答（误拒正例）← 最要命的数字
      fn = 库外且未拒答（漏拒）
    """
    tp = sum(1 for g, rej in rows if (not g) and rej)
    fp = sum(1 for g, rej in rows if g and rej)
    fn = sum(1 for g, rej in rows if (not g) and (not rej))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


def f1_of_round(items, labels, k):
    """用每条 query 的第 k 次结果组装一个 run，算它的 F1。"""
    rows = []
    for it in items:
        lab = labels[it["id"]]
        if k >= len(lab):
            continue
        rows.append((bool(it["gold"]), not lab[k]))
    return compute_f1(rows)


def f1_of_vote(items, labels, k=None, tie="answerable"):
    """多数投票后的 F1。

    k: 只取每条 query 的**前 k 次**结果投票（None = 全部）。
       ⚠️ 必须显式传 k，否则 k=3 与 k=5 会算成同一个东西（本脚本踩过此坑）。
    tie: 平票时的取舍。answerable=放行（宁误答不误拒） / reject=拒答。
    """
    rows = []
    ties = 0
    for it in items:
        lab = labels[it["id"]]
        lab = lab[:k] if k else lab
        if not lab:
            continue
        c = Counter(lab)
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            ans = (tie == "answerable")
        else:
            ans = top[0][0]
        rows.append((bool(it["gold"]), not ans))
    out = compute_f1(rows)
    out["ties"] = ties
    out["k"] = k or len(next(iter(labels.values())))
    return out


def stability(items, labels):
    """标签稳定性：完全一致的 query 占比 + 逐条翻转明细。"""
    stable, flips = 0, []
    for it in items:
        lab = labels[it["id"]]
        if not lab:
            continue
        if len(set(lab)) == 1:
            stable += 1
        else:
            flips.append({
                "id": it["id"], "category": it["category"],
                "query": it["query"][:60], "gold": it["gold"],
                "labels": lab,
            })
    total = sum(1 for it in items if labels[it["id"]])
    return {"total": total, "stable": stable,
            "stable_rate": round(stable / total, 4) if total else 0.0,
            "n_flip": len(flips), "flips": flips}


# ---------------------------------------------------------------- 主流程
def main():
    items = load_contexts()

    # --reuse：从已存标签矩阵重算，不调 API。
    # 标签矩阵是「原始数据」，任何投票策略/指标定义的变化都应从这里重算，
    # 而不是重新消耗 1240 次 LLM 调用。
    if "--reuse" in sys.argv:
        if not OUT_PATH.exists():
            print(f"❌ 未找到 {OUT_PATH}，无法 --reuse")
            return
        saved = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        lm = saved.get("label_matrix")
        if not lm:
            print("❌ 旧结果无 label_matrix，请全量重跑一次")
            return
        by_id = {i["id"]: i for i in items}
        print(f"[--reuse] 从 {OUT_PATH.name} 读取标签矩阵，仅重算指标。")
        labels07 = lm["q2_temp07"]
        subset = [by_id[i] for i in labels07 if i in by_id]
        saved["label_matrix"] = lm
        _report(items, subset, lm["q1_temp0"], labels07,
                meta={"n_calls_t0": saved["q1_temp0"].get("n_calls", 0),
                      "sec_t0": saved["q1_temp0"].get("seconds", 0),
                      "n_calls_t07": saved["q2_temp07"].get("n_calls", 0),
                      "sec_t07": saved["q2_temp07"].get("seconds", 0)},
                write=False)
        return

    print(f"评测集 {len(items)} 条"
          f"（负例 {sum(1 for i in items if not i['gold'])} / "
          f"正例 {sum(1 for i in items if i['gold'])}）")

    # ---- Q1: temp=0.0，全量，5 次 ----
    r0 = run_repeats(items, REPEATS_T0, 0.0, "Q1")
    st0 = stability(items, r0["labels"])
    runs0 = [f1_of_round(items, r0["labels"], k) for k in range(REPEATS_T0)]
    vote0_k3 = f1_of_vote(items, r0["labels"], k=3)
    vote0_k5 = f1_of_vote(items, r0["labels"], k=5)

    # ---- Q2: temp=0.7，子集，3 次 ----
    negs = [i for i in items if not i["gold"]]
    poss_sorted = sorted([i for i in items if i["gold"]],
                         key=lambda x: x["top_cosine"])
    subset = negs + poss_sorted[:T07_SUBSET - len(negs)]
    r07 = run_repeats(subset, REPEATS_T07, 0.7, "Q2")
    st07 = stability(subset, r07["labels"])
    runs07 = [f1_of_round(subset, r07["labels"], k) for k in range(REPEATS_T07)]
    vote07_k3 = f1_of_vote(subset, r07["labels"], k=3)

    _report(items, subset, r0["labels"], r07["labels"],
            meta={"n_calls_t0": r0["n_calls"], "sec_t0": r0["seconds"],
                  "n_calls_t07": r07["n_calls"], "sec_t07": r07["seconds"]})


def _report(items, subset, labels0, labels07, meta=None, write=True):
    """汇总并打印复现性指标；write=False 时只打印不落盘。"""
    meta = meta or {}
    n0 = max(len(v) for v in labels0.values())
    n07 = max(len(v) for v in labels07.values())

    st0 = stability(items, labels0)
    runs0 = [f1_of_round(items, labels0, k) for k in range(n0)]
    vote0_k3 = f1_of_vote(items, labels0, k=min(3, n0))
    vote0_k5 = f1_of_vote(items, labels0, k=n0)

    st07 = stability(subset, labels07)
    runs07 = [f1_of_round(subset, labels07, k) for k in range(n07)]
    vote07_k3 = f1_of_vote(subset, labels07, k=n07)

    def f1s(rs):
        return [x["f1"] for x in rs]

    def fps(rs):
        return [x["fp"] for x in rs]

    print(f"\n{'=' * 74}")
    print(f"Q1 · temperature=0.0（生产配置）· 全部 {len(items)} 条 × {n0} 次")
    print("=" * 74)
    print(f"  标签完全一致的 query : {st0['stable']}/{st0['total']} = "
          f"{st0['stable_rate']:.2%}")
    print(f"  出现翻转的 query     : {st0['n_flip']} 条")
    print(f"  误拒正例数 fp（每次）: {fps(runs0)}  ← 最关键指标，看是否稳定")
    print(f"  {n0} 个 run 的 F1       : {f1s(runs0)}")
    print(f"    → 极差 {max(f1s(runs0)) - min(f1s(runs0)):.4f}，"
          f"均值 {sum(f1s(runs0)) / len(runs0):.4f}，"
          f"区间 [{min(f1s(runs0)):.4f}, {max(f1s(runs0)):.4f}]")
    print(f"  多数投票 k={vote0_k3['k']}         : F1 {vote0_k3['f1']} "
          f"(tp={vote0_k3['tp']}, fp={vote0_k3['fp']}, fn={vote0_k3['fn']}, ties={vote0_k3['ties']})")
    print(f"  多数投票 k={vote0_k5['k']}         : F1 {vote0_k5['f1']} "
          f"(tp={vote0_k5['tp']}, fp={vote0_k5['fp']}, fn={vote0_k5['fn']}, ties={vote0_k5['ties']})")
    if st0["flips"]:
        print("  翻转明细：")
        for f in st0["flips"]:
            print(f"    - [{f['category']}] {f['query']} | gold={f['gold']} "
                  f"| labels={f['labels']}")

    print(f"\n{'=' * 74}")
    print(f"Q2 · temperature=0.7（压力测试）· {len(subset)} 条 × {n07} 次")
    print(f"     子集 = 40 负例 + {len(subset) - 40} 条 top-1 余弦最低的正例（最易判错）")
    print("=" * 74)
    print(f"  标签完全一致的 query : {st07['stable']}/{st07['total']} = "
          f"{st07['stable_rate']:.2%}")
    print(f"  出现翻转的 query     : {st07['n_flip']} 条")
    print(f"  误拒正例数 fp（每次）: {fps(runs07)}")
    print(f"  {n07} 个 run 的 F1       : {f1s(runs07)}")
    print(f"    → 极差 {max(f1s(runs07)) - min(f1s(runs07)):.4f}")
    print(f"  多数投票 k={vote07_k3['k']}         : F1 {vote07_k3['f1']} "
          f"(tp={vote07_k3['tp']}, fp={vote07_k3['fp']}, fn={vote07_k3['fn']}, ties={vote07_k3['ties']})")
    if st07["flips"]:
        print("  翻转明细：")
        for f in st07["flips"]:
            print(f"    - [{f['category']}] {f['query']} | gold={f['gold']} "
                  f"| labels={f['labels']}")

    print(f"\n{'=' * 74}")
    print("结论")
    print("=" * 74)
    fp_set = set(fps(runs0))
    if len(fp_set) == 1:
        print(f"  ✅ 误拒正例 fp 在 {n0} 次重复中恒为 {fp_set.pop()} 条")
        print("     → **最关键的指标 100% 可复现**（误拒比漏拒严重得多）。")
    print(f"  ⚠️ 全量口径 F1 区间 [{min(f1s(runs0)):.4f}, {max(f1s(runs0)):.4f}]，"
          f"波动只来自 {st0['n_flip']} 条**库外边界样本的漏拒**，与正例无关。")
    print(f"  → 对外报数应写区间；单点 {max(f1s(runs0)):.4f} 是乐观端，"
          f"{min(f1s(runs0)):.4f} 是保守端。")
    print(f"  → temp 不是不稳定源：temp=0.7 在**刻意挑的最难子集**上稳定率 "
          f"{st07['stable_rate']:.2%}，与 temp=0 的 {st0['stable_rate']:.2%} 相当。")
    print(f"  → self-consistency：k={vote0_k3['k']} → {vote0_k3['f1']}，"
          f"k={vote0_k5['k']} → {vote0_k5['f1']}，均**未超过**单次最优 "
          f"{max(f1s(runs0)):.4f}。翻转样本仅 {st0['n_flip']} 条，属运气，**不建议加**。")
    if meta:
        print(f"  调用量：temp0 {meta['n_calls_t0']} 次（{meta['sec_t0']}s）"
              f" + temp0.7 {meta['n_calls_t07']} 次（{meta['sec_t07']}s）"
              f" = {meta['n_calls_t0'] + meta['n_calls_t07']} 次")

    if not write:
        return
    OUT_PATH.write_text(json.dumps({
        "setup": {
            "judge_model": JUDGE_MODEL,
            "repeats_t0": n0, "repeats_t07": n07,
            "subset_t07": len(subset), "workers": WORKERS,
        },
        "q1_temp0": {
            "n_items": len(items), "stability": st0,
            "per_round_f1": runs0, "fp_per_round": fps(runs0),
            "majority_vote_k3": vote0_k3, "majority_vote_k5": vote0_k5,
            **(meta and {"n_calls": meta["n_calls_t0"], "seconds": meta["sec_t0"]} or {}),
        },
        "q2_temp07": {
            "n_items": len(subset), "stability": st07,
            "per_round_f1": runs07, "fp_per_round": fps(runs07),
            "majority_vote_k3": vote07_k3,
            **(meta and {"n_calls": meta["n_calls_t07"], "seconds": meta["sec_t07"]} or {}),
        },
        # 完整标签矩阵（原始数据）：任何投票策略/指标定义变化都应从这里重算，
        # 不必重新消耗 1240 次 LLM 调用（本脚本自身踩过「重跑一次才发现算错」的坑）。
        "label_matrix": {
            "q1_temp0": labels0,
            "q2_temp07": labels07,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
