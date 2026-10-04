# 数据集卡片（Dataset Cards）

> 本项目所有外部数据源的来源、许可、规模与采集记录。
> **合规纪律**：① 不对外分发原始数据 ② CC BY 4.0 需署名 ③ 每个数据集记录下载日期
> 更新日期：2026-09-28

---

## DC-01 · 3D Printing Troubleshooting Guide（排障知识库）

| 项 | 内容 |
|---|---|
| **用途** | RAG 知识库主源（problem → cause → solution） |
| **来源** | Mendeley Data · DOI `10.17632/88pcwt65yp.1`<br>GitHub: `hyungjungkim/3D-printing-troubleshooting-guide` |
| **贡献者** | Jingu Kang, Kangmin Ahn, Hyungjung Kim（Konkuk University，韩国建国大学） |
| **许可** | **CC BY 4.0**（需署名） |
| **发布** | 2025-12-30（Version 1） |
| **原始格式** | Markdown (233 KB) + PDF (267 KB) |
| **实测规模** | **75 个问题条目 / 314 条原因**（`# N.` 问题 + `## M)` 原因） |
| **上游数据源** | Simplify3D、Prusa3D Help、All3DP 三大排障文档，人工整理去冗 |
| **清洗记录** | 已解析为 `kb_v1.json`；标注 `is_fdm`（剔除 3 条非 FDM）、`img_classes`（22 条映射到图像类） |
| **已知问题** | ① 1 组完全重名（`Layer Shifting` ×2）② 大量同义条目（来自 3 个源）③ 2 条为纯叙述型无子标题（`Over-Extrusion`、`Curling or Rough Corners`）|
| **下载日期** | 2026-09-28 |

**引用要求**：Kang, J., Ahn, K., & Kim, H. (2025). 3D Printing Troubleshooting Guide. Mendeley Data, V1. doi:10.17632/88pcwt65yp.1

---

## DC-02 · Minds-R-Lab 3D-Printing-Dataset（🔴 2026-09-28 核查：**数据不可获取**，降级为"论文数字参照"）

| 项 | 内容 |
|---|---|
| **用途** | 原计划作为 CV 缺陷分类器训练主集（8 类 + Okay） |
| **来源** | GitHub: `Minds-R-Lab/3D-Printing-Dataset`（default branch `main`） |
| **许可** | ❌ **无 LICENSE 文件**（GitHub API 返回 `license: null`） |
| **标称规模** | README 声称 **2,638 张**，人工校验 |
| **类别** | Spaghetti / Stringing / Blobs-Zits / Warping / Z Seam / Layer Shifting / Delamination / Overhang + Okay |
| **采集条件** | iPhone 13 Pro，均匀光照，白底；Prusa MK4 + Prusa PLA + 默认 PrusaSlicer 设置；22–25°C、40–50% RH |
| **公布 baseline** | 10 个模型：ResNet18/34/50、VGG16、MobileNetV2、DenseNet121、EfficientNet-B0/B4、ViT-B/16、Swin-T<br>**Swin-T 最优：F1 87% / Acc 87%**；MobileNetV2 在 Spaghetti 类达 95% F1 |
| **⚠️ 注意** | 部分难以复现的类别来自外部数据集（如 Roboflow）→ 跨源一致性需抽查 |
| **下载日期** | ❌ 不可下载（见下） |

### 🔴 实测核查结论（GitHub API 直接验证，非推测）

**该仓库不包含数据集本体，且无任何下载入口。**

| 核查项 | 实测结果 |
|---|---|
| 仓库总条目 | **11 个**（`git/trees/main?recursive=1` 全量，`truncated: false`） |
| 文件构成 | 代码 5 个 `.py`（`config.py`/`dataset.py`/`models.py`/`trainpro.py`/`evaluatepro.py`）+ 配图 4 个（`Paper_Overview.jpg` + `figs/*.png`）+ `README.md` |
| 图片文件 | **0 张**（`figs/` 内是论文配图，不是数据集） |
| Releases | **0 个**（无 release asset） |
| 仓库体积 | 1,680 KB —— 2,638 张高分辨率图不可能塞进 1.7 MB |
| LICENSE | 无 |
| README 下载链接 | **无**（未指路 Google Drive / Zenodo / Kaggle） |

**代码暴露的数据约定**（说明数据原本只存在于作者本机）：

```python
# config.py
BASE_DIR = "F:/3dprinting"                      # ← 作者自己的 F: 盘
DATA_DIR  = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')     # torchvision ImageFolder 结构
VAL_DIR   = os.path.join(DATA_DIR, 'val')
CLASSES = ['Class 1 - Spaghetti', 'Class 2 - Stringing and oozing',
           'Class 3 - Blobs and Zits', 'Class 4 - Warping', 'Class 5 - Z Seam',
           'Class 6 - Layer Shifting', 'Class 7 - Layer Separation',
           'Class 8 - Overhang Defects', 'Class 9 - Okay']
```

**处置**：❌ **不作为数据源**。仅引用其**公开的 baseline 数字**（Swin-T F1 87%）作为学术对照——论文数字是公开的，可以用；数据拿不到，不能假装有。

---

## DC-02R · Kaggle `wengmhu` FDM 3D Printing Defect Dataset（✅ **替代 DC-02，P1 主数据源**）

| 项 | 内容 |
|---|---|
| **用途** | **P1 视觉缺陷分类主集**（替代不可获取的 DC-02） |
| **来源** | Kaggle `wengmhu/fdm-3d-printing-defect-dataset` |
| **许可** | ✅ **MIT**（可放心用于开源项目） |
| **规模** | **1,912 张**（Kaggle 包 6,293 MB，约 3.3 MB/张） |
| **类别（5 类，✅ 文件清单实拉核实）** | Warping **538** / Cracking **472** / Stringing **447** / Layer_shifting **364** / Off_platform **91**<br>合计 = **1,912** ✅ 自洽 |
| **热度** | 下载 2,231 / 13 票（同类检索结果中最高） |
| **最后更新** | 2024-10-17 |
| **核查方式** | 2026-09-28 经 Kaggle API 实测：`datasets/view` 得 `licenseName: MIT`；`datasets/list`（96 页分页全量遍历）得 1,912 个 `.jpg` 的真实类别分布 |
| **⚠️ 上游描述有笔误** | 数据集 README 写 `Cracking（427）`，**实为 472**（数字颠倒）。其自称总量 1,912 与分类相加 1,867 差 45，即由此笔误造成。<br>**以文件清单实测为准**：538+472+447+364+91 = 1,912，与总量完全吻合 |
| **引用要求** | MIT License, Copyright (c) 2024 Hu Wenjing, ChenChang<br>论文：Hu, W., Chen, C., Su, S. et al. *Real-time defect detection for FFF 3D printing using lightweight model deployment.* Int J Adv Manuf Technol **134**, 4871–4885 (2024). doi:10.1007/s00170-024-14452-4 |
| **目录结构** | `FDM-3D-Printing-Defect-Dataset/data/<类别>/Image_<时间戳>.jpg` |
| **⚠️ 注意** | 6.3 GB → **必须下载到 D 盘**（C 盘仅剩 13 GB） |

### 与知识库的类别映射（P1「视觉 → 知识库」映射层的直接依据）

| 数据集类别 | 张数 | 对应知识库条目 | 映射难度 |
|---|---|---|---|
| Stringing | 447 | KB006 `Stringing or Oozing` | ✅ 直接同名 |
| Warping | 538 | `Warping` | ✅ 直接同名 |
| Layer_shifting | 364 | `Layer Shifting` | ✅ 直接同名 |
| Cracking | 472 | 层分离 / 结构类 | ⚠️ 需逐条核对 |
| Off_platform | 91 | 附着类（首层不粘床等） | ⚠️ 需逐条核对 |

**采用理由**：① 许可唯一明确主流（MIT） ② 有真实图像 ③ 类内样本量均衡（91~538） ④ **3 类与知识库条目直接同名**，映射层不用硬凑。

### 检索记录（可复现）

```bash
# Kaggle API（凭据已存在于 ~/.kaggle/access_token）
GET https://www.kaggle.com/api/v1/datasets/list?search=3d%20printing%20defect&pageSize=12
GET https://www.kaggle.com/api/v1/datasets/view/wengmhu/fdm-3d-printing-defect-dataset
```

---

## DC-03 · Zenodo 19200472（图像备选 A）

| 项 | 内容 |
|---|---|
| **用途** | 翘曲/拉丝/开裂三类二分类补充 |
| **来源** | Zenodo `records/19200472` |
| **规模** | **11,593 张**高分辨率图 |
| **类别** | Warping (1,707) / No_Warping (1,815) / Stringing (2,555) / No_Stringing (1,811) / Cracking (1,663) / No_Cracking (2,042) |
| **采集条件** | Delta 打印机 + Raspberry Pi HQ Camera (12.3MP IMX477)；**PLA 与 ABS 双材料**；通过 DoE 系统变化喷嘴温度/热床温度/速度/冷却诱发缺陷 |
| **特色** | 含**无缺陷对照类**（适合做二分类与误报评估） |
| **下载日期** | 待下载 |

---

## DC-04 · Zenodo 18628777（✅ **2026-10-04 升格：P1 外部验证集 + 正常件门控数据源**）

| 项 | 内容 |
|---|---|
| **用途** | ⬆️ **2026-10-04 定位升级**：由「图像备选 B」升为 **P1 外部验证集**（跨数据集泛化证据）+ **正常/异常门控**数据源。**不替代 DC-02R 做主集**（无 Warping/Cracking/Off_platform，替换会断映射层） |
| **来源** | Zenodo `records/18628777`（Trier University of Applied Sciences / FH Aachen）<br>DOI `10.5281/zenodo.18628777`（v2.0.0，发布 2026-02-13） |
| **规模** | **48.4 GB**（additional_defect_data 16.1GB / base_data 19.6GB / test_data 12.6GB） |
| **类别** | 挤出异常（extrusion errors）/ 层错位（layershifts）/ 拉丝（stringing）/ **无缺陷（error-free）** |
| **特色** | **三个相机角度**采集；含**无缺陷对照件** → 可做「正常/异常」门控，避免把正常件硬判成某缺陷类 |
| **许可** | ✅ **CC BY 4.0**（2026-10-04 经论文参考文献交叉核实："License CC BY 4.0"）→ **必须署名** |
| **关联论文** | Bast S., Scherer K., Wahl M., Naumann S., Dartmann G., Didas S. *Reference-based monitoring for defect detection in fused filament fabrication using image feature classification.* **Int J Adv Manuf Technol** 144:2047–2059 (2026). DOI `10.1007/s00170-026-17844-w`（2026-04-10） |
| **该论文的公开 baseline** | 预训练 **ResNet-50 只做特征提取 + kNN**（不微调），各类 **F1 0.757~0.990，加权平均 0.812** → **P1 的L1 对照组即此方法**，口径可引用 |
| **代码** | `gitlab.rlp.net/ISS/realtime-defect-detection-in-additive-manufacturing` |
| **⚠️ 使用注意** | ① 体积大，**建议只下 `test_data.zip`（12.6 GB）**做外部验证；② **需核实是否有 part 级 ID**——若无，则同样存在近重复泄漏风险，须走 DC-02R 同一套 pHash+DINOv2 去泄漏流程 |
| **下载日期** | 待下载 |


---

## DC-05 · Cambridge Apollo（参数标签数据集）

| 项 | 内容 |
|---|---|
| **用途** | **参数补偿**能力（本项目 P2 决策段的关键） |
| **来源** | University of Cambridge Apollo `handle/1810/339869` |
| **许可** | **CC BY 4.0** |
| **规模** | **1,272,273 张**标注图像；含 948,396 张预过滤版本 |
| **标签** | **flow rate / lateral speed / Z offset / hotend temperature / hotend target temperature / bed temperature / timestamp / nozzle tip x,y** |
| **采集** | 8 台打印机机队自动采集，192 个不同零件 |
| **特色** | **唯一带完整打印参数标签的数据集** → 支撑"根因 → 参数调整"闭环 |
| **下载日期** | 待下载 |

**引用要求**：Brion & Pattinson. Generalisable 3D printing error detection and correction via multi-head neural networks. Apollo, University of Cambridge.

---

## DC-06 · 其他备选（2026-09-28 已逐个核验可达性）

| 数据集 | 来源 | 内容 | 实测状态 |
|---|---|---|---|
| 3D-Printed-or-Not | HF `cmudrc/3d-printed-or-not` | 二分类（是否 3D 打印件） | ✅ CC-BY-4.0 / 1–10 万张，但**任务是"识别是否打印件"而非缺陷分类** → 不匹配 |
| Failures in 3D printing | HF `Javiai/failures-3D-print` | 4 类（error/extrusor/part/spaghetti），**目标检测 bbox** | ⚠️ 仅 **73 张**、`license: unknown` → 规格不符 |
| 3D printing errors | HF `NilsHagenBeyer/3D_printing_errors` | 声明 apache-2.0 | ❌ **仓库内无数据文件**（只有 README + .gitattributes） |
| 3D-Printer Defected Dataset | Kaggle `justin900429/3d-printer-defected-dataset` | 缺陷图 | 需核验许可 |
| YOLO Stringing Dataset | Zenodo `records/14711320` | YOLO 格式拉丝检测 | 适合检测任务 |
| FDM Sensory Data | GitHub `adiparamtha/fdm_dataset` | **传感器时序数据** | 补"非图像通道" |
| 3D printing defects | Kaggle `ssharkov/3d-printing-defects` | 缺陷图 26.8 MB | ⚠️ 许可 Unknown |
| Defect-3D-Printing | Kaggle `muhammadmeftahmafazy/defect-3d-printing` | 201 MB，源自 Roboflow `fwe01/3d-printing-defect` | ⚠️ 许可 Unknown，需溯源 |
| 3D Printing Failure Detection | Kaggle `padraigvalenti/3d-printing-failure-detection` | 83.4 MB | ✅ **CC0 公共领域**（许可最宽松，可作补充） |

> 检索方式：Kaggle API `datasets/list?search=...` + HuggingFace `api/datasets?search=...`（2026-09-28 实测）。

---

## 相关学术工作（非数据，但必读）

| 工作 | 出处 | 为何重要 |
|---|---|---|
| **Keyword-RAG** | J. Comput. Des. Eng. 13(6):68-82, 2026<br>doi:10.1093/jcde/qwag052 | **用同一知识库做 FDM 排障 RAG**，report context recall **0.728** / precision **0.599**，对比 Naive-RAG 与 Rerank-RAG → 你的检索基线对标物 |
| LLM-3D Print | arXiv `2408.14307` | LLM 监控+控制 3D 打印，实时读参数并生成 G-code 修正 → **你的"决策-执行"段参照系** |
| Fault analysis in AM | JSS (2025) | ML + LLM 联合诊断，含 Prompt Evaluation Framework → 提示词策略参考 |

---

## 合规声明

1. 本项目数据**仅用于本地学习研究**，不对外分发原始数据
2. CC BY 4.0 数据集（DC-01 / DC-05）**必须署名**，成果中保留完整引用
3. 若项目开源，**只提交代码与索引脚本，不提交原始数据**；README 指路原始来源
4. **只采用许可明确的源**：DC-01（CC BY 4.0）、DC-02R（**MIT**）、DC-06 中的 CC0 项、DC-05（CC BY 4.0）。
| **DC-05**（CC BY 4.0）、**DC-04**（CC BY 4.0，2026-10-04 补登）。
   许可 Unknown 的一律**不入库**，除非能溯源到原始许可方
5. **DC-02 经核查数据不可获取且无许可** → 不作为数据源；仅引用其已公开发表的 baseline 数字（Swin-T F1 87%）
6. **引用他人数字必须标注出处**（2026-10-04）：Bast2026 的加权 F1 0.812、Swin-T 的 87%、植物病害研究的 +23.4pp 均为**他人结果**，引用时注明任务/数据集不同、不可直接比较
