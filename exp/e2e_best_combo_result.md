# 端到端最优组合验证：B2 kb 级索引 + qwen-plus + 一遍法

> 实验日期：2026-09-30｜脚本：`exp/exp_e2e_kb_onepass.py`｜数据：`exp/exp_e2e_kb_onepass_result.json`
> 对照基线：B2 两遍法来自 `exp/exp_e2e_kb_ab_result.json`（B 组）

## 一、四组实验汇总（端到端 strict 主指标）

| 组 | 索引粒度 | 生成法 | prompt | 模型 | strict | lenient |
|---|---|---|---|---|---|---|
| A | chunk 级 | 两遍法 | v2.2 | qwen-plus | 35.63% (57/160) | 61.25% |
| 历史基线 | chunk 级 | 一遍法 | v2.1 | qwen-plus | 39.37% (63/160) | 67.50% |
| B | B2 kb 级 | 两遍法 | v2.2 | qwen-plus | 43.75% (70/160) | 63.75% |
| **C（本实验）** | **B2 kb 级** | **一遍法** | **v2.2** | **qwen-plus** | **49.38% (79/160)** | **61.25%** |

## 二、核心结论

**最优组合 = B2 kb 级索引 + qwen-plus + 一遍法 = 49.38%（79/160）**，为当前全部实验的最高分。

- **「两遍法拖累」假设成立**：在严格同条件（v2.2 prompt + qwen-plus + B2 索引）下，
  一遍法 49.38% vs 两遍法 43.75% = **+5.63pp（控制变量，干净）**。
  两遍法在 chunk 级（A 35.63% < 历史一遍 39.37%）与 B2 级（B 43.75% < C 49.38%）两处都更差，
  说明 v2.2 实现的 Pass1 抽取是净拖累（过宽松喂噪声 / Pass2 丢 gold 根因），应弃用。
- **三层增益叠加**（相对最初 baseline chunk+turbo+v2.1 一遍 14.37%）：
  1. 生成模型 turbo→qwen-plus（chunk 级上 +11.25pp：28.12%→39.37%）
  2. 检索索引 chunk→B2 kb 级（gold@5 85%→90%，传导到生成 +5pp：43.75%→…）
  3. 关掉两遍法退回一遍法（B2 级 +5.63pp：43.75%→49.38%）

## 三、C 组分类别拆解（strict / lenient）

| 类别 | in_scope | correct | partial | wrong | strict 率 | lenient usable 率 |
|---|---|---|---|---|---|---|
| single_defect | 50 | 25 | 5 | 20 | 50.0% | 60.0% (30/50) |
| multi_defect | 30 | 10 | 12 | 8 | **33.3%** | 73.3% (22/30) |
| param_compensation | 40 | 22 | 4 | 14 | 55.0% | 65.0% (26/40) |
| ambiguous | 40 | 22 | 8 | 10 | 55.0% | 75.0% (30/40) |

- **multi_defect 仍最弱（33.3%）**：多缺陷题需同时穷尽多个 kb 的根因，B2 整条召回已解决检索覆盖，
  但生成层"穷尽多个 gold 根因"仍是瓶颈，是下一步主攻方向。
- lenient：C 61.25% 略低于 B 63.75%（−2.5pp）——一遍法 strict 更准但少了 Pass1 去噪，
  个别答案过生成从 usable 掉成 unusable。主指标 strict 净增 +5.63pp 已覆盖此小损。

## 四、⚠️ 重要更正：KB016 评测口径失真

此前（选项1 分析）我判断「KB016 空 kb 导致 eval_080 严格注定 0 分」——**此判断有误**。
实测 C 组 eval_080 = strict=correct。原因：KB016 的 `causes=[]`，
`build_reference(gold)` 生成的参照块**无根因条目**，严格判官按「覆盖参照中全部根因」判，
空参照 → 无根因可覆盖 → **trivially correct**（自动判对，非模型真能力）。

这暴露一个**评测口径缺陷**：空 kb 的 gold 题在严格判官下恒为 correct，既失真又掩盖知识库缺口。
正确处理（择一）：① 给 KB016 补 2–3 条真实根因；② 把 eval_080 gold 指向描述 curling 的真实含根因 kb；
③ 判官特判空 kb 为 invalid。当前 79 个 correct 中含此 1 题失真项，真实正确数 ≈ 78/159 ≈ 49.06%，影响可忽略但须标注。

## 五、下一步（按优先级）

1. **接入主检索路（推荐）**：确认最优组合后，改 `HybridRetriever` 支持 kb_chunks.json（B2），
   重建 faiss+bm25 落盘；生成器改 `TWO_PASS` 默认关（或删两遍法代码）。主指标预期 43.75%→49.38%。
2. **补 KB016 根因**：消除评测失真 + 提升真实系统覆盖。
3. **攻 multi_defect（33%）**：两遍法已弃用，可试「枚举纪律 prompt」（强制逐条列出每个命中 kb 的全部根因）
   或答案后处理校验 gold 覆盖率。
4. **提交 git**：本次所有 exp 文件（exp_e2e_kb_ab*.py / result / onepass / index_*.md）尚未提交，
   坐标已澄清——发布仓本地 main 停在 v2.2，改动请基于含 v2.4/v2.5 的分支。

## 六、成本（呼应额度顾虑）

付费 embedding 模式：B2 索引 56 块 + 200 query ≈ ¥0.04；qwen-plus 生成 ~200 次 ≈ ¥3；
判官 qwen-turbo 很便宜。整场 ≈ ¥3–4。embedding/rerank 额度几乎不动（rerank 未用）。
