# 选项③验证：结构化「完整性回填」攻击生成层瓶颈

> 实验日期：2026-09-30｜脚本：`exp/exp_e2e_kb_backfill.py`｜数据：`exp/exp_e2e_kb_backfill_result.json`
> 分析：`exp/analyze_backfill.py`
> 基线：B2 kb 级索引 + qwen-plus + 一遍法(v2.2)，post-KB016-fix 重建索引

## 一、瓶颈定位（为什么是选项③）

端到端 160 题 in_scope 未拒答，gold 全部进 top-5 上下文的有 **131 题**：
- 其中 77 correct / 21 partial / 33 wrong
- **gold 已在上下文却没答对 = 54 题**（21 partial + 33 wrong）= 生成层瓶颈，检索侧已尽力
- 剩下 29 题 gold 未全进 top-5 = 检索缺口，回填帮不了

这 54 题就是选项③的可攻击目标。

## 二、方法（受控消融，不碰生产 src）

对基线 v2.2 一遍法生成的草拟答案，仅在「gold⊆检索 top-5 且 strict∈{partial,wrong}」的题上加 **Call2 完整性回填**：
- 输入只有检索到的证据（与运行时可用信息一致，**不注入 gold**，诚实）
- 指令：审查草拟答案是否覆盖证据中**全部相关**根因；只补「证据里已有、且确实相关、但答案漏掉」的根因；严禁补证据外内容
- 对回填后答案重新判官
- 下行有界：回填只可能「多加噪声」不会「丢失已答对的 gold」；correct 题不动

## 三、结果

| 口径 | 基线(v2.2一遍法) | +回填 | Δ |
|---|---|---|---|
| strict 成功 | 77/160 = 48.13% | **86/160 = 53.75%** | **+5.62pp（+9 correct）** |
| lenient usable | 112/160 = 70.0% | 112/160 = 70.0% | 0 |

**回归：0**（strict 0 例 correct→非correct；lenient 0 例 usable→unusable）。

### 回填题（37 题实际被 augmented）转化矩阵
| 基线→处理 | 题数 |
|---|---|
| wrong → correct | 9 |
| wrong → partial | 2 |
| wrong → wrong | 13 |
| partial → partial | 13 |

（54 个目标中，17 题回填判定 add=[] 未改动；37 题补了内容，其中 11 题改善）

### 按类别 strict 成功数（基线 → 处理）
| 类别 | 基线 | 处理 |
|---|---|---|
| single_defect | 25/50 | **30/50** |
| multi_defect | 11/30 | 13/30 |
| ambiguous | 21/40 | 23/40 |
| param_compensation | 20/40 | 20/40 |

## 四、⚠️ 口径更正（重要，防误报）

`onepass` 原报 79/160=49.38% 是 **KB016 补根因(feaff08)之前**的 B2 索引数字。
本次实验在 KB016 补根因、索引重建**之后**跑，重算同索引基线 = **77/160**。
8 题因 KB016 现在进 top-5 改变了检索而 verdict 变动（索引重建效应，**非回填效应**）。
故诚实对比应为：同索引下 **77 → 86（+5.62pp）**，回填是干净增益。

## 五、结论

选项③（生成后完整性回填）**有效且零回归**，是当前最大单点增益（+5.62pp），直接把
"gold 在上下文却答错"的 54 题中的 11 题救回（9 wrong→correct + 2 wrong→partial）。
新 SOTA = **B2 + qwen-plus + 一遍法 + 回填 = 86/160 = 53.75%**。

## 六、剩余天花板与下一步

- **检索缺口 29 题**：gold 没进 top-5，回填无解 → 需选项②(top-k=10) 或检索再提精度。
- **回填漏判 17+13 题**：模型判官认为"草拟已完整"或回填补了仍判 wrong（草拟有错误框架/噪声，回填加正确内容也救不回）→ 需更强回填相关性判别或答案重写而非追加。
- **生产化待办**：实验用"仅非 correct 才回填"模拟理想；真实系统不知 gold，需对**每题**都跑回填 → 须先验证"对 correct 题回填不回归"再接入主路。确认后把回填折成 `answer_generator.py` 的 v2.4 方法并定版、push GitHub。

## 七、生产化验证（回填折进 src 主路，全量零回归）✅

> 脚本：`exp/exp_e2e_kb_backfill_prod.py` + `exp/analyze_backfill_prod.py`；数据：`exp/exp_e2e_kb_backfill_prod_result.json`
> 回填已从 exp 内联副本**折进 `src/answer_generator.py` 的 `generate_with_backfill`**，`agent_graph` 的 generate 节点切到它，`config.ANSWER_BACKFILL` 默认开（每题都跑，生产现实）。
> 本验证在同一次运行内对每题算 baseline(关回填) 与 treatment(开回填)，用同一套集成代码、同一 B2 索引，杜绝跨文件口径漂移。

| 口径 | 基线(纯一遍法) | +回填(全开) | Δ |
|---|---|---|---|
| strict 成功 | 77/160 = 48.13% | **87/160 = 54.37%** | **+6.25pp（+10 correct）** |
| lenient usable | 99/160 = 61.88% | **111/160 = 69.37%** | **+7.5pp** |
| **回归** | — | **strict 0 / lenient 0** | — |

- **零回归安全门通过**：回填全开（含本来答对的 77 题）下，correct→非 correct = 0、usable→unusable = 0；77 答对题中 22 道被回填补充仍 correct、55 道判无需补原样保留。回填实际触发 73/160。
- 转化矩阵(回填题)：correct→correct 22 / wrong→wrong 20 / partial→partial 19 / **wrong→correct 10** / wrong→partial 2。
- 按类别 strict(基线→处理)：single 25→30 / multi 11→13 / ambiguous 21→23 / param 20→21。
- **新 SOTA = 54.37%（生产级，post-KB016-fix 同索引）**；六章的 86/160=53.75% 是"仅非 correct 才回填"的理想模拟，真实全开为 87/160=54.37%，更优。
- **本地提交 commit `53d161d`**（src 3 文件 + exp 脚本/报告/结果）；push 因代理 502 瞬时故障待恢复后重推。
