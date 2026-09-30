# KB016 知识库补全 + 评测失真消除（2026-09-30）

## 0. 背景（失真是什么）
`KB016` = "Curling or Rough Corners"，原 `causes=[]`、`cause_count=0`（空 kb）。
`eval_080`（ambiguous / spec_negation，query="bed temp correct but corners curl"，gold=["KB016"]）在严格判官下：
- 判官调用 `build_reference(gold)` 把 `KB016.causes=[]` 拼成 **「只有问题、无根因」的空参照**
- 严格判官按「覆盖参照**全部**根因」判分 → 空参照无可覆盖 → **trivially correct（自动判对）**
- 这是**评测口径失真**：eval_080 的 correct 不是模型能力，是空参照漏洞

→ 之前 C 组（B2 kb 级 + qwen-plus + 一遍法）报 79/160=49.38%，**含此 1 题失真项**。

## 1. 修复（已落地）
给 `kb_v2.json` 的 KB016 补 **3 条真实根因**（英文，与全库一致），正对 eval_080 语义
（床温正确但角落卷 → 是附着/冷却/收缩，而非床温）：
1. **Poor first-layer adhesion**（首层附着不足，即使床温对也卷）
2. **Insufficient part cooling**（层间冷却不足，后期卷）
3. **High material shrinkage**（材料收缩率，ABS/ASA 上翘）

`cause_count` 0→3。重建 B2 索引（`build_kb_index.py`，KB016 块现含 3 根因）。

## 2. 验证结果（exp/verify_kb016.py，独立新缓存强制重判）
| 项 | 补前 | 补后 |
|---|---|---|
| KB016 根因数 | 0 | 3 |
| eval_080 严格 verdict | correct（失真） | **partial**（真实） |
| 判官参照 | 空（无根因） | 含 3 条真实根因 |

**检索侧**：eval_080 top-5 = [KB016, KB024, KB101, KB044, KB037]，KB016 排第一 ✅

**⚠️ 但模型未采纳 KB016**：答案 17 条对策**全部基于 KB024**（"该证据与问题关联性弱，未采纳：[2] KB016, retrieved"）。
即 eval_080 的真实能力本就该靠 KB016，模型却偏好 KB024（更泛的翘曲条目）。

## 3. 结论
- **失真消除目标达成** ✅：KB016 不再是空 kb，评测能真实判分。
- **真实 strict 修正**：79/160 → **78/160 = 48.75%**（去掉 1 题失真；48.75% 比 49.38% 更可信）。其余 159 题不受 KB016 块变动影响。
- **暴露更深问题（待攻）**：模型在检索已召回 KB016(top-1) 时仍不采纳、改引 KB024 → **证据采纳/选择**层短板，属"多候选 kb 时模型偏好长块/泛化条目"的老难题，与索引粒度无关，需 prompt 层约束（强制采纳 top-1 证据 + 按 gold 根因措辞）或 KB016/KB024 内容去重。

## 4. 交付物
- `data/kb_v2.json`：KB016 补 3 根因（已 commit）
- `data/kb_chunks.json` + `faiss_index_kb.bin` + `bm25_index_kb.pkl`：重建含新根因（已 commit）
- `exp/verify_kb016.py`：失真消除验证脚本（独立缓存强制重判）
- `exp/_kb016_verify_cache.json`：验证判官缓存（新建）

⚠️ 旧 `data/judge_e2e_cache.json` / `exp_e2e_kb_onepass_result.json` 中 eval_080 仍为失真 correct，
全量数字须重跑（或仅重判 eval_080）才反映真实 78/160；单题已验证真实=partial。
