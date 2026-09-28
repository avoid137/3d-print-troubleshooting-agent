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

## DC-02 · Minds-R-Lab 3D-Printing-Dataset（图像主集）

| 项 | 内容 |
|---|---|
| **用途** | CV 缺陷分类器训练/测试（8 类 + Okay） |
| **来源** | GitHub: `Minds-R-Lab/3D-Printing-Dataset` |
| **许可** | GitHub 公开仓库（使用前请再次核验仓库 LICENSE） |
| **规模** | **2,638 张**，人工校验 |
| **类别** | Spaghetti / Stringing / Blobs-Zits / Warping / Z Seam / Layer Shifting / Delamination / Overhang + Okay |
| **采集条件** | iPhone 13 Pro，均匀光照，白底；Prusa MK4 + Prusa PLA + 默认 PrusaSlicer 设置；22–25°C、40–50% RH |
| **已提供 baseline** | 10 个模型：ResNet18/34/50、VGG16、MobileNetV2、DenseNet121、EfficientNet-B0/B4、ViT-B/16、Swin-T<br>**Swin-T 最优：F1 87% / Acc 87%**；MobileNetV2 在 Spaghetti 类达 95% F1 |
| **⚠️ 注意** | 部分难以复现的类别**来自外部数据集（如 Roboflow）** → **跨源类别一致性需抽查** |
| **下载日期** | 待下载 |

**可对比基线**：直接用其公布的 Swin-T 结果作为你的对照，避免"自己报自己"。

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

## DC-04 · Zenodo 18628777（图像备选 B）

| 项 | 内容 |
|---|---|
| **用途** | 多角度、含无缺陷对照的大规模补充 |
| **来源** | Zenodo `records/18628777`（Trier University of Applied Sciences） |
| **规模** | **48.4 GB**（additional_defect_data 16.1GB / base_data 19.6GB / test_data 12.6GB） |
| **类别** | 挤出异常、层错位、拉丝 + 无缺陷 |
| **特色** | **三个相机角度**采集 |
| **⚠️ 注意** | 体积大，按需部分下载 |
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

## DC-06 · 其他备选

| 数据集 | 来源 | 内容 | 备注 |
|---|---|---|---|
| 3D-Printer Defected Dataset | Kaggle `justin900429/3d-printer-defected-dataset` | 缺陷图 | 需核验许可 |
| Failures in 3D printing | HF `Javiai/failures-3D-print` | 4 类（error/extrusor/part/spaghetti），**目标检测 bbox 格式** | 适合检测任务 |
| YOLO Stringing Dataset | Zenodo `records/14711320` | YOLO 格式拉丝检测 | 适合检测任务 |
| FDM Sensory Data | GitHub `adiparamartha/fdm_dataset` | **传感器时序数据** | 补"非图像通道" |

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
4. 使用 DC-02 前**再次核验其 LICENSE**
