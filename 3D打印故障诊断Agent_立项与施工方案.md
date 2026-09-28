# 3D 打印故障诊断 Agent · 立项与施工方案

> ⚠️ **本文档为 v0 立项稿（2026-09-28 19:20），部分内容已过时，仅作历史留档。**
>
> 与现行口径不一致之处（**以 `00_项目准备总览.md` 为准**）：
>
> | 项 | v0 立项稿 | 现行 |
> |---|---|---|
> | 图像数据集 | Minds-R-Lab（2,638 张 / 8 类） | **已换源**：Kaggle `wengmhu`（MIT，1,912 张 / 5 类）；原数据集**经核查不可获取且无 LICENSE** |
> | P1 分类数 | 8 类 | **5 类** |
> | 图像验收线 | 对标官方 Swin-T 87% | 自建 ResNet-34 baseline，≥ 85%（**不同数据集不可同集对比**） |
> | 拒答机制 | 阈值法（拟） | **LLM 判官**，F1 0.9487（区间 [0.9351, 0.9487]） |
> | 检索口径 | 待定 | Recall@5 = **85.00%**；端到端有效回答率 **85.00%** |
>
> 版本演进详见 `01_版本修改记录_v1-v2.md`。

> 立项日期：2026-09-28 ｜ 定位：**Agent 系统架构级项目**（对标 Multi-Agent / Skills / Harness / 自优化 四主线 JD）
> 前置核验：知识库与图像数据集的**类别对齐已实测完成**，见 §3

---

## 0. 一页纸总览

| 项 | 定案 |
|---|---|
| **方向** | 3D 打印（FDM）故障诊断与质量优化 Agent |
| **对标 JD** | 3D 打印排障 Multi-Agent 架构岗（Skills / Harness / 自优化三条主线） |
| **知识库** | 3D Printing Troubleshooting Guide（CC BY 4.0）→ **75 问题 / 314 原因**，已清洗为 `kb_v1.json` |
| **图像数据** | Minds-R-Lab 3D-Printing-Dataset（2,638 张 / 8 类缺陷 + Okay） |
| **存储** | PostgreSQL + pgvector（知识向量）+ 关系表（参数/案例） |
| **编排** | LangGraph：感知 → 诊断 → 决策 → 执行 |
| **工期** | 4 周，P0/P1/P2/P3 四阶段 |
| **差异化** | ① 缺陷分类映射层 ② Skills 化诊断能力 ③ 评测驱动的 Harness 自优化外循环 |

---

## 1. 为什么是这个方向

### 1.1 JD 要求与现有能力的真实对照

| JD 要求 | 你的现状 | 判定 |
|---|---|---|
| Multi-Agent + LangGraph/AutoGen 状态编排 | 单 Agent 挂多工具 | ❌ 缺口 |
| **Agent Skills 模块化能力工程** | 7 工具 + `prompts/*.yaml` + `loader.py`（注册/校验/动态加载） | ✅ **可直接转化** |
| **Harness 工程化治理** | DSML 状态机 / 9 态会话 / `session_scope` / fail-fast 双策略 | ✅ **可直接转化** |
| RAG + 向量库 | 仅离线消融脚本 | ❌ 缺口 |
| 自优化 / MCP / AI Coding | 无 | ❌ 缺口（但差异化最大） |
| AIOps / 工业故障诊断经验 | 无 | ❌ 缺口 |

**核心判断**：你不是从 0 开始，是**有零件但没装配成 JD 要的那台机器**。

### 1.2 为什么选 3D 打印

| 理由 | 说明 |
|---|---|
| **数据真实可得** | 知识库与图像数据集均已公开、许可清晰（见 §2） |
| **故障可复现** | 翘曲/拉丝/层错位都能通过改参数人为复现 → **有物理真值** |
| **JD 明确优先** | "有 3D 打印、增材制造或相关硬件领域知识者优先" |
| **可对接硬件** | OctoPrint / Marlin 有成熟 API，参数补偿能真正执行（呼应 JD"深度集成打印机传感、控制 API"） |

### 1.3 学术前沿已确认（也是差异化压力）

**Keyword-RAG**（J. Comput. Des. Eng., 2026）已用同一份排障知识做 FDM 排障 RAG，report 出 context recall 0.728 / precision 0.599。

→ **意义**：赛道被确认有效，但**基础版 RAG 已不新鲜**。你的差异化必须落在 **Agent 编排 + Skills + 自优化**——**这恰好是 JD 要的**，学术论文反而帮你锁定了发力点。

---

## 2. 数据源清单（已核验）

| 用途 | 数据集 | 规模 | 许可 | 来源 |
|---|---|---|---|---|
| **知识库** | 3D Printing Troubleshooting Guide | 75 问题 / 314 原因 | **CC BY 4.0** | [Mendeley](https://data.mendeley.com/datasets/88pcwt65yp) ｜ [GitHub](https://github.com/hyungjungkim/3D-printing-troubleshooting-guide) |
| **图像主集** | Minds-R-Lab 3D-Printing-Dataset | 2,638 张 / 8 类 + Okay | GitHub 公开 | [GitHub](https://github.com/Minds-R-Lab/3D-Printing-Dataset) |
| **图像备选** | Zenodo 19200472 | 11,593 张 / 翘曲·拉丝·开裂 | Zenodo | [Zenodo](https://zenodo.org/records/19200472) |
| **图像备选** | Zenodo 18628777 | 48.4 GB / 含无缺陷对照 | Zenodo | [Zenodo](https://zenodo.org/records/18628777) |
| **参数标签** | Cambridge Apollo | 127 万张 / **含流率·温度·Z-offset 标签** | **CC BY 4.0** | [Apollo](https://www.repository.cam.ac.uk/handle/1810/339869) |
| 补充 | Kaggle justin900429 / HF `Javiai/failures-3D-print` | — | 各自 | — |

**主组合**：Minds-R-Lab（8 类、已人工校验、有 10 模型 baseline 可对比）+ Cambridge（参数标签，支撑参数补偿）。

⚠️ **合规纪律**：每个数据集在 README 建"数据集卡片"（来源 / 许可 / 规模 / 采集日期）；不对外分发原始数据；CC BY 4.0 需署名。

---

## 3. 知识库 ↔ 图像对齐（**已完成实测**）

### 3.1 覆盖矩阵

| 图像缺陷类（英文 / 中文） | 知识条目 | 原因数 | 状态 |
|---|---|---|---|
| Spaghetti / 空中乱丝 | 1 | 2 | ⚠️ **薄弱** |
| Stringing / 拉丝渗料 | 2 | 15 | ✅ |
| Blobs-Zits / 表面疙瘩 | 2 | 8 | ✅ |
| Warping / 翘曲 | 4 | 15 | ✅ 强 |
| Z Seam / 层接缝 | 2 | 7 | ✅ 名称需映射 |
| Layer Shifting / 层错位 | 5 | 25 | ✅ 很强 |
| Delamination / 层间分离 | 3 | 10 | ✅ |
| Overhang / 悬垂桥接 | 3 | 14 | ✅ |

**结论：8 类全部有知识支撑，无一落空；但有 3 处必须处理。**

### 3.2 三个必须处理的问题

**① Spaghetti 覆盖薄弱（唯一硬缺口）**
图像有该类，知识库仅 1 条间接条目（"Stringy and Droopy"，且它归在 Over-Extrusion 下）。
**Spaghetti 与 Stringing 成因不同**（前者是模型脱离支撑在空中挤出，后者是回抽不足），硬映射会答错。
→ **对策**：从 All3DP / Simplify3D 原站补 2–3 条 Spaghetti 专属条目。

**② 命名体系不一致 → 需手写映射层**
图像用**视觉术语**（Z Seam / Delamination），知识库用**现象描述**（Diagonal Scars / Layer Separation）。两边不是自动能连上。
→ **成果**：`mapping.yml` 作为独立交付物（面试可讲"数据对齐"）。

**③ 非 FDM 污染 + 重复条目**
- **3 条非 FDM**：`Print doesn't appear / Resin does not solidify`、`Resin too Cold`、`Underpowered Laser` → **剔除**
- **1 组完全重名**：`Layer Shifting` × 2（来自不同源）
- **大量同义条目**：如 `Warping` vs `Print Edges are Bending (Warping)`；`Under-Extrusion` vs `Print Layers Look Uniformly Thin/Weak`
→ **对策**：去重合并，产出"原始 75 条 → 去重后 N 条"的数字

### 3.3 另一个方向的失衡（可讲的设计洞察）

**知识库有、但图像数据集无对应类的条目：53 条**（Under-Extrusion 14 条、Clog/Nozzle 系列、First Layer/Adhesion 系列、Vibration/Ringing 等）。

→ **含义**：知识库的覆盖面**远大于**图像数据集。这说明**不能只用视觉分类驱动诊断**——"喷嘴堵塞""首层不粘"这类故障在图上不明显，需要**传感器数据 + 用户描述**补充。
→ **这正是 Multi-Agent 存在的理由**：单一视觉通道不够，必须多源融合。

---

## 4. 已产出数据文件

| 文件 | 内容 |
|---|---|
| `data/kb_v1.json` | 清洗后知识库：75 条，字段 `kb_id / problem / is_fdm / img_classes / intro / causes[{cause, solution}]`，共 **314 条原因**，22 条已建立图像类映射 |
| `data/mapping.yml` | 8 类图像缺陷 ↔ 知识条目 ID 映射（含待补项） |
| `data/dataset_cards.md` | 数据集卡片（来源/许可/规模/采集日期） |

---

## 5. 系统架构

```
用户输入（缺陷照片 + 现象描述）
        │
   ┌────▼─────┐
   │ 感知 Agent │  CV 分类器（8 类 + Okay）＋ 传感器/描述解析
   └────┬─────┘
        │  缺陷类别 + 置信度
   ┌────▼─────┐
   │ 诊断 Agent │  RAG 检索知识库 → 根因候选排序（引用溯源）
   └────┬─────┘
        │  根因 + 证据链
   ┌────▼─────┐
   │ 决策 Agent │  参数补偿方案生成（约束校验 + 安全边界）
   └────┬─────┘
        │  参数调整建议 / G-code
   ┌────▼─────┐
   │ 执行 Agent │  下发 OctoPrint / Marlin API（模拟环境）
   └────┬─────┘
        │
   评测闭环 → Harness 自优化外循环（回写提示词/工具定义）
```

**四段与 JD 一一对应**：JD 原文即"感知—诊断—决策—执行"。

### 5.1 存储分工

| 数据 | 存储 | 理由 |
|---|---|---|
| 排障知识（problem/cause/solution） | **pgvector** | RAG 检索，约 300–500 chunk |
| 缺陷图像 | 文件系统 + 关系表索引 | 走 CV 模型，**不进向量库** |
| 参数-缺陷对照 | **关系表** | 结构化，SQL 查询 |
| 故障案例历史 | 关系表（+ 相似检索可选） | 复盘与自优化依据 |

⚠️ **诚实边界**：几百 chunk 用 pgvector 或 FAISS 都够，**不要吹"处理百万向量"**。README 写明升级阈值（chunk > 10 万 / QPS 高 → Milvus）。

---

## 6. Agent Skills 设计（JD 第 2 条核心）

**复用你已有的 loader.py 范式**（注册 / 校验 / 动态加载）：

| Skill | 输入 | 能力 | 依赖 |
|---|---|---|---|
| `diagnose_stringing` | 图像 + 参数 | 拉丝根因排序 | KB 检索 |
| `diagnose_warping` | 图像 + 材料/环境 | 翘曲根因（含热应力） | KB + 材料表 |
| `diagnose_layer_shift` | 图像 + 机械参数 | 层错位根因（皮带/加速度） | KB |
| `diagnose_clog` | 现象描述 + 传感器 | 喷嘴堵塞推理 | KB（53 条无图类） |
| `compensate_params` | 根因 + 当前参数 | 参数补偿建议 | 约束校验 |
| `verify_improvement` | 前后对比 | 改善验证 | 图像对比 |

**Skills 注册表**（`skills/registry.yml`）：声明 name / 触发条件 / 输入 schema / 依赖 / 版本。
**动态加载**：按诊断阶段按需加载，而非全量注入提示词（这正是"通用 Agent 按需获取专业技能"）。

---

## 7. 评测集设计（核心资产）

**规模 200 条，五类**：

| 类别 | 占比 | 说明 |
|---|---|---|
| 单缺陷诊断 | 25% | 图像 → 类别 → 根因 |
| 多缺陷并发 | 15% | 一张图多种缺陷，考优先级 |
| 需参数补偿 | 20% | 根因 → 可执行参数调整 |
| **库外/罕见故障** | **20%** | 知识库无对应，**正确行为是说明不确定 + 建议人工** |
| 诱导 / 歧义 | 20% | 相似缺陷混淆（Spaghetti vs Stringing 必考） |

**六项指标**：

| 指标 | 定义 |
|---|---|
| 缺陷分类准确率 | 8 类 + Okay 混淆矩阵 |
| 根因 Top-3 命中率 | 正确根因是否在前三 |
| **拒答/存疑准确率** | 库外题正确标记不确定 ← 最难也最值钱 |
| 参数建议合理率 | 是否在安全边界内 |
| 引用命中率 | 引用的 KB 条目是否真支撑结论 |
| 端到端通过率 | 全链路正确 |

**判分纪律（沿用无人超市已验证方法）**：`run_eval.py` 只跑不判；`score_eval.py` 只判不跑；**判分器不调 LLM**；判据人工复核一轮。

---

## 8. Harness 自优化外循环（JD 第 3 条，最值钱）

**你的独有优势**：已有 `run_eval`/`score_eval` 闭环，别人要从零搭评测才能谈自优化。

```
① 跑评测（200 条）
      ↓
② 按失败类型归因（检索未召回 / 根因排序错 / 参数越界 / 该拒答未拒答）
      ↓
③ 自动改写 Harness 组件（提示词 / Skill 描述 / 检索参数 k / 重排权重）
      ↓
④ 重跑评测 → 与历史版本对比
      ↓
⑤ 择优保留（存档 harness_v{N}.yml，记录因果：改了哪项 → 哪个指标动了）
```

**产出**：一张"Harness 版本演进表"——**第 N 版改了什么、哪项指标涨了多少**。这是 JD 要的"通过历史 Harness 的因果推理实现架构级自主改进"的**最小可行实现**。

⚠️ **诚信边界**：若只实现"提示词自动改写 + 择优"，就说这个，**不要宣称做了完整的 Meta-Harness 架构自进化**。

---

## 9. 四阶段里程碑

### P0（第 1 周）— 数据与最小闭环
- 知识库清洗入库（去重、剔非 FDM、建映射）→ 向量库
- 图像数据集下载 + CV 分类器 baseline
- **验收**：CV 8 类分类准确率 ≥ 85%（对标 Swin-T 的 87%）；RAG 检索 Recall@5 ≥ 0.70（对标 Keyword-RAG 0.728）

### P1（第 2 周）— 单 Agent 诊断链
- 感知 → 诊断两段打通；引用溯源
- 评测集 100 条（含负例）
- **验收**：根因 Top-3 命中率 ≥ 70%；拒答准确率 ≥ 80%

### P2（第 3 周）— Multi-Agent + Skills
- LangGraph 四段编排；Skills 注册与动态加载
- 决策 + 执行段（OctoPrint 模拟环境）
- **验收**：端到端通过率 ≥ 60%；Skills 单元测试全绿

### P3（第 4 周）— 自优化 + 交付
- Harness 外循环；评测集扩到 200 条
- Docker Compose + README + 技术报告
- **验收**：出 Harness 版本演进表（至少 3 版，展示指标变化）；报告含失败案例 ≥ 5 例

---

## 10. 明确不做

| 不做 | 理由 |
|---|---|
| 完整 Meta-Harness 架构自进化 | 学术前沿，4 周做不完；做"提示词+检索参数"版即可 |
| 自训练 CV 大模型 | 用现成 backbone 微调（MobileNetV2 / Swin-T）足够，有 baseline 可对标 |
| 真实打印机硬件对接 | 无设备；用 OctoPrint 模拟环境，README 说明边界 |
| Milvus / Kafka / K8s | 规模不需要，会被问穿 |
| 覆盖全部 75 类故障 | 聚焦 8 类图类 + 高频无图类，**做透 > 做全** |

---

## 11. 简历写法（口径与无人超市对齐）

**项目名**：3D 打印故障诊断与参数优化 Agent

**一句话**：面向 FDM 3D 打印的故障诊断 Multi-Agent 系统。LangGraph 编排"感知—诊断—决策—执行"四段；自建 75 条排障知识库（314 条根因-对策）与 8 类缺陷图像分类器；诊断能力封装为可动态加载的 Skills；内置 200 条评测集（含 20% 库外负例），并以评测驱动 Harness 提示词级自优化。

**技术栈**：LangGraph · FastAPI · PostgreSQL + pgvector · DashScope Embedding · PyTorch(分类) · Vue3 · Docker Compose

**三条要点（数字留空，做完填真值）**：
1. 缺陷分类：8 类 + Okay，准确率 __%（baseline Swin-T 87%）
2. 诊断：根因 Top-3 命中率 __%，拒答准确率 __%
3. 自优化：Harness 迭代 __ 版，某指标从 __% 提升至 __%

⚠️ **诚信约束**：数字**做完才填**；未完成期注明"持续迭代中"；不使用"Meta-Harness 架构自进化"等超出实现的表述。

---

## 12. 风险与待核实

| # | 事项 | 状态 |
|---|---|---|
| 1 | **Spaghetti 知识缺口**需从 All3DP/Simplify3D 补 2–3 条 | 待做 |
| 2 | 图像数据集**跨源类别一致性**（官方声明部分类来自 Roboflow）需抽查 | 待做 |
| 3 | 精确率/召回对标 Keyword-RAG(0.728/0.599)——需在同一知识库上复现才能对比 | 待确认 |
| 4 | 本机**无 Docker / 无 PostgreSQL**（已实测）→ 需先装，pgvector 走 Windows 原生 DLL 路线 | 硬约束 |
| 5 | 本机 **GPU 情况未知**，CV 分类器微调可行性待确认 | 待确认 |
| 6 | 工期 vs 10 月秋招密度 | 需排期 |

---

## 13. 一句话总纲

> 这个项目的价值不在"做了个 3D 打印诊断"，而在**用真实数据把 Agent 的四条工程主线（编排 / Skills / Harness / 自优化）全部走通一遍**——而且每条都能从无人超市找到可复用的零件。
