# 端到端配对验证：B2 kb 级索引 是否把检索增益传导到生成层

> 实验脚本：`exp/exp_e2e_kb_ab.py`（配对 A/B，控制生成方法不变，只换检索索引）
> 运行条件：qwen-plus + 两遍法固定；A=chunk 级（生产现状，gold@5=85%），B=B2 kb 级（cause+solution 全量，56 块，gold@5=90%）
> 全量 200 题，判官 qwen-turbo（strict v1 + lenient v1），~1h37m
> ⚠️ 付费 embedding 模式已开（用完即停关闭后跑通）

## 一、总览（控制变量：生成=两遍法/qwen-plus）

| 口径 | A chunk 级（控制） | B B2 kb 级（实验） | 差值 |
|---|---|---|---|
| in-scope 题数 | 160 | 160 | — |
| **strict correct** | 57 (35.63%) | 70 (43.75%) | **+8.12pp** |
| strict partial | 75 | 42 | −33 |
| strict wrong | 28 | 48 | +20 |
| **lenient usable** | 98 (61.25%) | 102 (63.75%) | **+2.50pp** |
| 负例误放行（应拒答却答） | 0/40 | 0/40 | 安全 |

**结论：检索增益确实传导到了生成层**——B2 把端到端 strict 从 35.63% 拉到 43.75%（+8.12pp），lenient +2.5pp，且不破坏拒答。

## 二、用户最关心的两个问题：partial→correct 是否真增？kb 块更长噪声多大？

**转移矩阵（A→B，160 in-scope 题）：**

| 转移 | 数量 | 性质 |
|---|---|---|
| partial→correct | **27** | ✅ 增益（kb 整条召回补齐了缺失根因） |
| wrong→correct | 5 | ✅ 增益 |
| wrong→partial | 5 | ✅ 增益 |
| correct→correct | 38 | 持平 |
| partial→partial | 29 | 持平 |
| partial→wrong | 19 | ⚠️ 退化（噪声） |
| correct→wrong | 11 | ⚠️ 退化（噪声） |
| correct→partial | 8 | ⚠️ 退化 |
| wrong→wrong | 18 | 持平 |

- **partial→correct = 27**（用户担心的"穷尽已存在 gold 根因"正是这 27 题被救回）→ 增益真实，不是幻觉。
- **噪声真实存在**：38 题退化（19 partial→wrong + 11 correct→wrong + 8 correct→partial），主要因 kb 块更长（KB002 等 18 根因块 ~3000–4000 字）稀释了模型焦点。
- **净增 correct = +13**（70−57）：增益 32（27+5）− 退化 19（11+8）= +13，与总表吻合。

## 三、分类别 strict correct（B−A）

| 类别 | A | B | B−A |
|---|---|---|---|
| single_defect | 19/50 | 25/50 | **+6** |
| multi_defect | 4/30 | 9/30 | **+5** |
| param_compensation | 14/40 | 17/40 | **+3** |
| ambiguous | 20/40 | 19/40 | −1 |

- 多缺陷 / 单缺陷 / 参数补偿 三类均受益（大 kb 不再内耗槽位），与检索层 gold@5 观察一致。
- **ambiguous 微降 1 题**：合理——ambiguous 是 query 歧义问题，与检索粒度无关，B2 无额外红利甚至略引入噪声。

## 四、⚠️ 必须上报的旁挂发现：两遍法可能反而拖累

本实验 A 组（chunk 级 + 两遍法）strict = **35.63%**，但历史基线（一遍法 qwen-plus）实测是 **39.37%**（commit 335a0d0）。

即：**两遍法（v2.2 实现）在当前评测下比一遍法低 ~3.7pp**。可能原因（待查）：
1. Pass1 抽取"宁可多留"过于宽松，喂给 Pass2 的候选含噪声，反而稀释；
2. 或本实验 harness（GraphDiagnoseAgent）与历史 exp_e2e_eval.py 不同，存在 prompt/证据格式差异（非纯两遍法效应）。

**推论**：B2 的 +8pp 增益是相对"两遍法控制组"的；若两遍法确实拖累，则 **B2 + 一遍法** 可能比 B2 + 两遍法更高（~47%+）。需一次确认性实验：B2 索引 + qwen-plus **一遍法** 跑全量，定位最优组合。

## 五、下一步建议（按优先级）

1. **确认最优组合（推荐）**：B2 kb 级 + qwen-plus 一遍法 跑全量，确认是否 > 43.75%（验证"两遍法拖累"假设）。
2. **降噪声（若保留两遍法）**：B2 块过长导致 correct→wrong/partial→wrong，可在两遍法 Pass1 抽取后做"块内根因裁剪"或截断超长 kb 块，压住退化。
3. **接入主检索路**：确认最优组合后，改 `HybridRetriever` 支持 kb_chunks.json，把胜出索引替换进生产（需重建 faiss+bm25 落盘）。
4. **补 KB016**：eval_080（ambiguous，gold=KB016）因空 kb 严格注定 0 分，补 2–3 条真实根因可救回 1 题。

## 六、原始数据
- 明细：`exp/exp_e2e_kb_ab_result.json`（200 题逐条 A/B 判定）
- 脚本：`exp/exp_e2e_kb_ab.py`（已修汇总 KeyError 容错）
