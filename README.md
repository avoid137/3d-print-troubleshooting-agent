# 3D 打印故障诊断 Agent · P0 检索层

> ## 🔴 换电脑后先读这里
>
> **`换机交接.md`** —— 环境重建、数据补齐、10 分钟验证清单、
> 对外口径（最容易讲错的一句话）、D10 现状、下一步优先级。
>
> ⚠️ 2026-10-07：GitHub 在当前网络下被**域名级阻断**，
> **本地有 27 个提交未推送** + tag `backup-20261007`。
> 换机后第一件事 = `git push origin main --tags`（先清掉旧代理配置）。

面向 FDM 3D 打印排障场景的**知识检索与根因定位**系统。

当前进度：**P0 阶段完成** —— 知识库清洗、切分、BM25 + 向量双路检索、RRF 融合、200 条评测集。
P1 视觉分类已接通（详见 `HANDOFF_P1.md`）。

---

## 🔴 对外口径（唯一正确说法）

> 在 **DC-02R 数据集内**，采用 **group-aware折外交叉验证**，
> 5 类缺陷 **macro-F1 = 99.73%**。

❌ 不能说「识别准确率 99.7%」/「SOTA」/「能识别任意打印机的缺陷」
——🔴 换机器实测只有 **6.25%**（DC-04外部验证，低于随机 20%）。

---

## 实测指标

| 指标 | 数值 | 说明 |
|---|---|---|
| **Recall@5**（检索口径，不含拒答） | **85.00%**（136/160） | 200 条评测集中的 160 条正例 |
| single_defect | 80.00% | 单缺陷定位 |
| multi_defect | 86.67% | 多缺陷组合 |
| param_compensation | 82.50% | 需参数补偿 |
| ambiguous | 92.50% | 诱导歧义 |
| **库外拒答（两层：预过滤 + LLM 判官）** | **F1 = 1.0000** | 库外正确拒答 **40/40** / **误拒率 0.00%** |
| 　└ 仅 LLM 判官层 | F1 = **0.9873** | 库外 39/40 / 误拒 0/160 |

**端到端效果**（160 条库内正例，分母统一，三项之和 = 1.0000）：

| 结果 | 本方案（两层拒答） | 余弦阈值法 |
|---|---|---|
| **有效回答率（答对）** | **85.00%** | 46.88% |
| 误答率（自信答错） | 15.00% | 8.13% |
| 误拒率（该答没答） | **0.00%** | 45.00% |

> 检索上限 = 136/160 = **85.00%**。本方案做到 **85.00%**，也就是**一条检索命中都没被浪费**——
> 拒答机制零误拒、零损耗。

**消融对比**：

| 方案 | Recall@5 |
|---|---|
| BM25 单路 | 78.12% |
| 向量单路（text-embedding-v3） | 75.00% |
| **Hybrid（0.6 : 0.4）** | **85.00%** |

> **方法论要点**：向量单路（75.00%）**低于** BM25 单路（78.12%），但融合后达 85.00%（**+6.88pp**）。
> 原因在于两路是否**异质**——BM25 与 TF-IDF 同属词袋模型，融合**零增量**
> （hybrid 78.12% == bm25 单路 78.12%）；BM25 与语义 embedding 异质，融合产生互补（+6.88pp）。
>
> 权重扫描（0.3~0.7，步长 0.1）：83.75 / 84.38 / 84.38 / **85.00（0.6:0.4）** / 83.75 —— 单峰，最优仍是 0.6:0.4。

**拒答方案对比**（40 条库外负例 / 160 条正例）：

| 方案 | 库外拒答率 | **误拒正例率** | Precision | F1 |
|---|---|---|---|---|
| 余弦阈值（0.6865） | 92.50% | **45.00%** ❌ | 33.94% | 0.4966 |
| LLM 判官单层（prompt v8） | 97.50% | **0.00%** | 100.00% | 0.9873 |
| **两层：预过滤 + LLM 判官** | **100.00%** | **0.00%** | **100.00%** | **1.0000** |

**两层设计**（`src/rejector.py`）：

```
Query → 检索 → ┌ Layer 1 · 确定性预过滤（纯规则、零 API、完全可复现）
               │   用户在索取「资料/指南/标准数据」→ 直接拒答
               └ Layer 2 · LLM 范围判官（两步判据：先判意图、再判现象）
```

> **为什么要有 Layer 1**：LLM 判官在「公差/规格」类问题上会被**话题牵引**——
> `tolerance chart` → 「公差」→「尺寸偏差」→ 误判库内。这个失败在**两份不同集子上各出现 1 次**
> （属稳定缺陷而非随机波动），改 prompt 措辞 3 轮均未修好，于是下沉为确定性规则。
>
> ⚠️ Layer 1 是关键词规则，**存在误伤风险**（若某库内问题碰巧含 `chart` / `table` / `manual` 等词）。
> 已做误伤扫描：在 3 份集子（200 + 40 + 24）上命中 7 条**全部为负例、正例误伤 0 条**。
> 该层可用 `rejector.PREFILTER_ENABLED = False` 一键关闭。

> ⚠️ **度量口径的两个坑**（`src/evaluate_reject.py` 头部有详细注释）
> ① 分母用「被放行的正例数」→ 拒答越激进数字越虚高（幸存者偏差，阈值法能被刷到 81.82%）；
> ② 分子用「检索命中」而不扣除被拒答条目 → 该指标与拒答机制无关，任何方案都得同一个数。
> 正解：分子 =「未拒答 且 命中」，分母 = 全部正例。

**判官可复现性**（`exp/exp_judge_reproducibility.py`，测的是**LLM 层单独**的表现）：

| 实验 | 规模 | 标签一致率 | 误拒正例数（每轮） | F1 |
|---|---|---|---|---|
| temp=0.0（生产配置） | 200 条 × 5 次 | **200/200 = 100.00%** | **恒为 0 条** | **0.9873（零波动）** |
| temp=0.7（压力测试） | 80 条 × 3 次 | **80/80 = 100.00%** | 恒为 0 条 | 0.9873（零波动） |

> **prompt 版本演进带来的一个意外收获**：v2 时代（单层判据「话题是否在范围内」）一致率 99.00%、
> F1 区间 [0.9351, 0.9487]（宽 0.0136）；改成 **v8 两步判据**后，**翻转归零、方差归零**。
> 结论：**不稳定不是模型的锅，是判据模糊**——边界样本上模型自己在两种解读间摇摆，
> 把决策路径写明确（先判意图、再判现象）之后，方差自然消失。
>
> 多数投票（k=3 / k=5）结果与单次相同（0.9873），**不建议为此加 self-consistency**。

**留出集验证**（`exp/exp_holdout_scope.py`）——因为 prompt 是在评测集上迭代出来的，必须防过拟合：

| 集子 | 规模 | 准确率 | 库内放行 | 库外拒答 |
|---|---|---|---|---|
| 留出集 #1 | 40 条（手写） | **40/40 = 100.00%** | 20/20 | 20/20 |
| 留出集 #2 | 24 条（手写） | **24/24 = 100.00%** | 12/12 | 12/12 |

> ⚠️ **诚实的局限**：两份留出集都是**同一位作者在规则设计之后**手写的，
> 只能证明「规则不是只对原本那 40 条负例生效」，**不能完全排除作者偏见**。
> 严格的泛化证据需要**第三方标注**或真实的线上 query 日志。

---

## 快速开始

### 1. 环境

```bash
# 依赖：faiss-cpu / scikit-learn / rank_bm25 / numpy / dashscope / chromadb
pip install faiss-cpu scikit-learn rank_bm25 numpy dashscope chromadb
```

### 2. 配置密钥（可选但推荐）

```bash
cp .env.example .env
# 编辑 .env，填入阿里云百炼（DashScope）的 API Key
```

**不配 Key 也能跑**：向量路会自动降级为 TF-IDF + SVD 离线兜底
（hybrid Recall@5 = 78.12%，**与 BM25 单路持平**——同质方法融合零增量；权重需改回 `0.5/0.5`）。

### 3. 跑起来

```bash
cd src

python chunking.py        # 知识库 → chunk（371 个）
python bm25_index.py      # 构建 BM25 索引
python vector_index.py    # 构建 FAISS 向量索引
python evaluate.py        # 评测，输出 Recall@5
python ablation.py        # 消融：bm25 / vector / hybrid 对比
python tune_weights.py    # 权重扫描
python evaluate_reject.py # 拒答评测（需 Key，含 200 次判官调用）
python evaluate_reject.py --reuse   # 仅重算指标，不重新检索/不调 API
```

复现性 & 留出集验证（`exp/`，需 Key）：

```bash
cd exp
python exp_judge_reproducibility.py          # 复现性：200×5 + 80×3，约 1240 次调用 / 100 秒
python exp_judge_reproducibility.py --reuse  # 从已存标签矩阵重算，不调 API
python exp_holdout_scope.py                  # 留出集 #1（40 条）
python exp_holdout_scope.py --set 2          # 留出集 #2（24 条）
python exp_chroma_eval.py                     # Chroma 向量路 ≡ FAISS 等价验证（无 Key 跑 TF-IDF 后端）
```

---

## 目录结构

```
.
├── data/
│   ├── kb_v1.json              原始知识库（75 条问题 / 314 原因）
│   ├── kb_v2.json              清洗后（56 条 / 315 原因，去重 + Spaghetti 补充）
│   ├── chunks.json             切分结果（371 chunk）
│   ├── eval_set.json           200 条评测集
│   ├── reject_eval_result*.json 拒答评测明细（含逐条理由）
│   ├── judge_repro_result.json 复现性实验标签矩阵（可离线重算）
│   ├── holdout_scope_result*.json 留出集验证明细
│   └── dataset_cards.md        数据来源、许可、采集说明
├── src/
│   ├── config.py               全局配置（含 .env 加载）
│   ├── preprocess.py           kb_v1 → kb_v2 清洗
│   ├── chunking.py             problem 级 + cause 级切分
│   ├── bm25_index.py           BM25 索引
│   ├── vector_index.py         FAISS 向量索引（DashScope / TF-IDF 双模式）
│   ├── chroma_index.py         **Chroma 元数据过滤补充层**（FAISS 之上的按需过滤，复用 vector_index 的 embedding）
│   ├── hybrid_retriever.py     RRF 双路融合检索
│   ├── build_eval_set.py       评测集构造
│   ├── evaluate.py             检索评测（Recall@5）
│   ├── ablation.py             消融实验
│   ├── tune_weights.py         权重调优
│   ├── rejector.py             两层拒答（预过滤 + LLM 判官）
│   └── evaluate_reject.py      拒答评测
├── exp/                        实验脚本（embedding 对比 / 复现性 / 留出集 / Chroma 等价验证）
├── .env.example                密钥模板
└── .gitignore                  保护 .env + 排除索引产物
```

---

## 系统设计

**双路检索 + RRF 融合**：

```
Query
  ├─→ BM25（词频，精确术语敏感）
  └─→ 向量检索（DashScope text-embedding-v3，1024 维，语义）
        ↓
    RRF 融合（Reciprocal Rank Fusion, K=60）
        ↓
    Top-5 chunk
```

**为什么用 RRF 而不是加权分数**：BM25 分数量纲（无上界）与余弦相似度（0~1）不可比，RRF 只看排名、对分数尺度不敏感，是 hybrid search 的工业标准做法。

**切分策略**：每个 problem 生成 1 个 problem 级 chunk，每个 cause 生成 1 个 cause 级 chunk，共 371 个。

**向量库定位（三套，职责不重叠）**：

| 向量库 | 产物 | 角色 | 何时用 |
|---|---|---|---|
| **FAISS**（`faiss_index.bin` / `faiss_index.tfidf_svd.pkl`） | 主向量索引 | **P0 正典向量检索**（hybrid 的向量路） | 生产检索、评测 |
| **Chroma**（`data/chroma_db/`，git 忽略） | 元数据过滤层 | **FAISS 之上的按需补充层**：在向量召回结果上按 `type` / `kb_id` 等元数据过滤，或 `where_document $contains` 关键词精筛 | 需要「向量 + 结构化过滤」组合的子场景 |

> ⚠️ **Chroma 不是 FAISS 的替代品**，而是补充：它复用 `vector_index` 的同一套 embedding，且向量 L2 归一化后 **cosine（Chroma）≡ 内积（FAISS `IndexFlatIP`）**，
> 因此纯向量召回率**与 FAISS 逐项一致**（实测 Chroma 117/160 = 0.7312 == FAISS-TF-IDF）。Chroma 的增量价值在于**元数据过滤**，而非召回质量。
> 验证脚本：`exp/exp_chroma_eval.py`（无 Key 跑 TF-IDF 后端，配 Key 可验 DashScope 后端应得 75.00%）。

---

## 评测集设计

200 条，五类均匀分布：

| 类别 | 条数 | 考察点 |
|---|---|---|
| single_defect | 50 | 基础定位 |
| multi_defect | 30 | 组合定位 |
| param_compensation | 40 | 参数补偿场景 |
| **out_of_scope** | **40** | **库外拒答（核心资产）** |
| ambiguous | 40 | 抗诱导 |

每条字段为 **6 个**：`id` / `query` / `category` / `gold` / `source` / `gold_basis`。

| 字段 | 含义 | 取值 |
|---|---|---|
| `gold` | 正确答案的 kb_id 列表（负例为 `[]`） | kb_v2 编号 |
| `source` | **这条 query 是怎么造出来的** | `kb_paraphrase` 50 / `kb_combination` 30 / `spec_param` 40 / `spec_negation` 40 / `out_of_scope` 40 |
| `gold_basis` | **gold 照着哪些 kb 条目出的**（出题依据） | 如 `KB003+KB004`；受过编号规范化的写成 `KB040 → KB012`（共 14 条） |

**构建可复现**：`python src/build_eval_set.py` 固定 `random.seed(42)`，
重跑输出与入库文件 **md5 逐字节一致**（`e7dceee2…`），且构建期有 `gold ⊆ kb_v2 ids` 断言拦截悬空引用。

---

## 已知局限

1. ~~拒答机制失效~~ → ✅ **已修复**
   原 RRF 分数与余弦分数均无法用阈值切分（分离间隔 -0.1865）。
   现为**两层拒答**：确定性预过滤（资料类请求）+ LLM 两步判官。
   库外正确拒答 40/40、误拒率 0.00%、F1 1.0000（LLM 层单独 0.9873）。

2. ~~embedding 方法无校验~~ → ✅ **已修复**
   `faiss_index.meta.json` 升级为 manifest，记录 `method` / `model` / `dim` / `count` / `built_at`；
   `HybridRetriever` 启动时校验，不一致直接抛错并给出修复命令。
   > 值得注意：DashScope 与 TF-IDF+SVD 的维度**恰好都是 1024**，
   > 所以方法不匹配时 FAISS **不会报错**，只会静默返回无意义的相似度——
   > 这比抛异常危险得多，必须主动拦截。

3. **评测集 gold 曾有 7 个悬空引用**（✅ 已修）— 实测发现
   建集脚本写于 kb_v1（75 条）时代，而清洗出 kb_v2（56 条）时按 `preprocess.py::MERGE_GROUPS`
   合并了同义条目、被合并的次 id 被删除，建集脚本却没同步 →
   `eval_set.json` 里 7 个 gold（`KB039/040/045/057/059/064/066`）指向**不存在的 kb_id**，
   涉及 **14 条题**，判分 `any(kb in gold for kb in retrieved_kb_ids)` 永远为 False
   → **这 14 条结构性判不了命中**，旧口径 127/160 因而是被低估的。
   ✅ **修法**：映射**不靠人工推断**，直接由 `kb_v2.json` 的 `merged_from` 字段派生
   （与清洗逻辑同源）；规范化后 14 条中 12 条命中恢复，主口径 **127/160 → 136/160 = 85.00%**。
   同时在 `build_eval_set.py` 加了 `gold ⊆ kb_v2 ids` 断言，**防止复发**。
   > ⚠️ 注意：修 `build_eval_set.py` 时**不要动 `SINGLE_DEFECT_QUERIES` 的 key**（其中含 v1 编号 `KB040`），
   > 否则 `defect_ids` 变化会让 `random.shuffle` 的配对全变、破坏与历史结果的可比性——统一在生成后规范化。

4. **`source` 有两层含义，只补齐了其中一层**（✅ 评测集那层已补）

   为避免混淆，先说清是两个不同的 `source`：

   | 位置 | 含义 | 现状 |
   |---|---|---|
   | `eval_set.json` 每条 | **这条 query 是怎么造出来的**（`source`）+ **gold 照着哪条 kb 出的**（`gold_basis`） | ✅ **已补齐**（200/200） |
   | `kb_v2.json` 每条 | **该知识条目的原始出处**（哪份上游文档） | ⚠️ 仅 **3/56**（3 条 Spaghetti 补录项带 `source`；其余 53 条出处集中记在 `data/dataset_cards.md`） |

   `eval_set.json` 现在每条长这样（`gold_basis` 里出现 `→` 表示该题受过 v1→v2 编号规范化）：

   ```json
   {"id": "eval_021", "query": "...", "category": "single_defect",
    "gold": ["KB012"], "source": "kb_paraphrase", "gold_basis": "KB040 → KB012"}
   ```

   这一层补上后有两个直接收益：① 能回答「这 200 条谁标的、依据什么」；
   ② 具备在构建期自动校验引用有效性的机制（第 3 条那个缺陷正是缺这层机制才漏了两个版本）。
   > 注意：新增字段是**纯追加**，`evaluate.py` / `evaluate_reject.py` 只读 `gold`，无需任何改动。

   ⚠️ **仍是缺口的**：`kb_v2.json` 条目自身的出处只有 3/56 条落在字段里，
   其余靠 `dataset_cards.md` 集中说明。若要做到**逐条可追溯**，需补 53 条 —— 属独立事项，未做。

5. **数据规模有限**
   知识库 56 条问题 / 315 条原因，覆盖 5 类常见 FDM 缺陷（挤出 / 附着 / 表面 / 结构 / 几何）；库外问题无法回答（这也是拒答机制的由来）。

6. **拒答的泛化证据仍不充分**
   prompt 是在同一份 200 条评测集上迭代（v2 → v8）出来的，**存在过拟合风险**。
   已用两份手写留出集（40 + 24 条）交叉验证（均 100%），但同作者手写不能完全排除偏见。
   下一步应引入**第三方标注**或**真实 query 日志**。

7. **Layer 1 预过滤是关键词规则**
   若某库内问题碰巧含 `chart` / `table` / `manual` 等词，会被误拒（当前 3 份集子上误伤 0 条）。
   可用 `rejector.PREFILTER_ENABLED = False` 关闭。

---

## 数据来源与许可

| 数据 | 来源 | 许可 |
|---|---|---|
| 知识库 | [3D-printing-troubleshooting-guide](https://github.com/hyungjungkim/3D-printing-troubleshooting-guide) | CC BY 4.0 |
| Spaghetti 补充条目 | All3DP / Simplify3D 公开文档 | 引用需注明 |

详见 `data/dataset_cards.md`。

⚠️ 原始数据集不随仓库分发；索引产物（`*.bin` / `*.pkl`）已通过 `.gitignore` 排除，可本地一键重建。

---

## 下一步

- [x] ~~拒答机制重做~~ ✅ **已完成**（两层：预过滤 + LLM 判官，F1 **1.0000**）
- [x] ~~`meta.json` 记录 embedding 方法并加校验~~ ✅ **已完成**（manifest + 启动校验）
- [x] ~~端到端指标口径修正~~ ✅ **已完成**（有效回答率 **85.00%**，零误拒，打满检索上限）
- [x] ~~判官 prompt 复现性验证~~ ✅ **已完成**（一致率 **100.00%**，F1 零波动 0.9873）
- [x] ~~拒答边界样本收口~~ ✅ **已完成**（prompt v2→v8 两步判据；方差从 ±0.0136 归零）
- [x] ~~留出集交叉验证~~ ✅ **已完成**（40 + 24 条手写集均 100%）
- [x] ~~评测集 gold 悬空引用修正~~ ✅ **已完成**（7 个 kb_v1 编号规范化为 kb_v2，14 条题恢复；主口径 127/160 → **136/160 = 85.00%**）
- [x] ~~评测集补 `source` 字段~~ ✅ **已完成**（200 条全部含 `source` 构造类型 + `gold_basis` 出题依据；14 条带 v1→v2 映射标记）
- [x] ~~Chroma 补充过滤层 + 等价验证~~ ✅ **已完成**（`src/chroma_index.py` 元数据过滤层；`exp/exp_chroma_eval.py` 证向量路 ≡ FAISS，117/160 = 0.7312）
- [ ] `kb_v2.json` 逐条补原始出处 `source`（当前 3/56，其余 53 条记于 `dataset_cards.md`）
- [ ] 拒答的**第三方标注**泛化验证（当前留出集为同作者手写，存偏见风险）
- [ ] P1：接入 5 类缺陷图像分类（Kaggle `wengmhu`，MIT，1,912 张），构建视觉 → 知识库映射层
