# kb 级 vs chunk 级 索引 A/B 对比（检索层 Recall@5）

> 实验日期：2026-09-30 ｜ 口径：HybridRetriever RRF(0.6:0.4, K=60) + top-5，复用 src/evaluate.py 按返回单元 kb_id 是否覆盖 gold 判命中。
> 脚本：exp/exp_index_ab.py（独立，不动原 chunks.json / 评测集）。

## 基线校验
A 组（chunk 级现状）复现 **136/160 = 0.8500**，与项目基线一致 → 实验口径正确。

## 汇总（160 正例 Recall@5）
| 配置 | Recall@5 | 命中/总 | 较现状 |
|---|---|---|---|
| A chunk 级（现状） | 0.8500 | 136/160 | — |
| B1 kb 级（仅 cause 短语） | 0.8750 | 140/160 | +2.50pp |
| B2 kb 级（cause+solution 全量） | 0.9000 | 144/160 | **+5.00pp** |

## 各类别（B2 vs A）
| 类别 | A | B2 | 变化 |
|---|---|---|---|
| single_defect | 0.8000 | 0.8200 | +0.02 |
| multi_defect | 0.8667 | 0.9667 | **+0.10** |
| param_compensation | 0.8250 | 0.9250 | **+0.10** |
| ambiguous | 0.9250 | 0.9250 | 0（无关） |

## 拒答区分度（负例 top-1 RRF 分）
A 0.0158 / B1 0.0162 / B2 0.0162，均极低 → kb 级未引入误放行风险。

## 结论
kb 级索引（B2 全量）全面优于 chunk 级：检索 Recall@5 **+5pp**，multi/param 类（大 kb 多根因场景）收益最大，验证「chunk 级粒度错配」论点；single 不退化、ambiguous 无关；负例区分度保持。
**推翻此前「检索侧到顶（仅剩生成侧可攻）」的判断**——索引粒度本身是检索增益来源。

## 下一步
用 B2 kb 级索引跑端到端 strict（qwen-plus + 两遍法），验证生成层 partial→correct 是否真增（kb 块更长，须防噪声；~40min LLM）。
