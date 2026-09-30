# 选项②验证：top-k=10 —— 死路（负结果，已证伪）

> 实验日期：2026-09-30｜检索分析：`exp/exp_topk10_analysis.py`｜定向生成实验：`exp/exp_e2e_topk10_focused.py`
> 数据：`exp/exp_e2e_topk10_focused_result.json`

## 一、检索覆盖（零 LLM 开销）
B2 kb 级索引，top-k 从 5 提到 10：
- gold⊆top5 覆盖 = **125/160 = 78.1%**
- gold⊆top10 覆盖 = **141/160 = 88.1%**
- k=5 缺口 35 题 → k=10 救回 **16 题**进上下文（single 4 / multi 5 / ambiguous 4 / param 3）；仍 19 题 k=10 也救不回。

## 二、定向生成实验（仅 16 道 k=10 才进上下文的题）
对这 16 题用 k=10 检索 + 生成 + 回填，对比其 k=5+回填 基线 verdict：

| 题数 | k5 基线 | k10 | 说明 |
|---|---|---|---|
| 11 | wrong | wrong | 未改善 |
| 1 | partial | partial | 持平 |
| 1 | correct | **partial** | **退化**（稀释） |
| 3 | correct | correct | 持平 |
| 0 | 非correct | correct | **上行 = 0** |

**净增 correct = -1**（k10 仅 3 correct vs k5 的 4 correct）。

## 三、结论
- **top-k=10 对 strict 准确率零增益、且有退化风险**：gold 进上下文后，生成层（一遍法+回填）仍不穷尽这些根因（与 v2.3 两遍法失败同源——生成层「穷尽已存在 gold 根因」的缺口），而多塞 5 块噪声稀释了原本答对的题（eval_044 correct→partial）。
- **与 rerank 同为检索侧死路**：检索召回已够（90%@5），剩余 ~31pp 缺口在生成层，检索精度/广度提升均不传导。
- **判定：选项② 否决**。不要再烧 ~90min 跑全量 k=10（上行 0、下行靠稀释，净必为负或持平）。
- 剩余严格增益只能来自**生成层重构**（答案重写 / 结构化穷尽，非追加式回填；但 v2.3 两遍法已证净负，需新方法）。当前 SOTA 维持 **54.37%**（B2+qwen-plus+一遍法+回填，零回归）。
