# 3D 打印故障诊断 Agent · 项目记忆（仓库自动加载）

> 随仓库 clone。**权威完整文档 = `HANDOFF_P1.md`**；所有实测数字明细 = `exp/p1_vision_result.md`。
> 本文件只留「不能忘的」，细节请回查上面两份。逐日过程在 `.workbuddy/memory/YYYY-MM-DD.md`。

## 状态一句话

P0 文本诊断已交付并**验证到顶**（strict SOTA 54.37%，别再碰检索/生成层）。
P1 视觉分类已跑完并可信（99.73% macro-F1），映射层看图定稿 5/5，端到端 + 拒答已通，
**P0↔P1 接线已打通**（`skill_vision_diagnose.py` + `agent_vision.py`，P0 冻结区零改动）。
**唯一未闭合风险 = DC-04 外部验证卡网络**（Zenodo 16 KB/s）。

## 纪律红线（6 条，不可违背）

1. **数字做完才填**，禁止预估。
2. **P1 不改 P0 检索/生成层**（已证重做必回归）。
3. **原始数据不进 git**；权重/模型/缓存全 gitignore。
4. **许可明确才用**（DC-02R MIT、DC-04 CC BY 4.0 需署名）。
5. **映射不硬凑，必须先看图**（见下面教训②）。
6. **对外口径**：99.73% 只能说「DC-02R 内、group-aware 折、5 类 macro-F1」，
   **不能**说「缺陷识别准确率 99.7%」或「SOTA」。

## ✅ P1 结论（数字勿改）

选定 **L2 冻结骨干 + 线性探针**（EfficientNet-B0 99.73±0.06% / MobileNetV3-small 99.41±0.70%）。
L1 98.4% / L3 99.19%（**无增益，别再跑 L3**）。两骨干等价（差 0.06~0.31pp）
→ 已触及「ImageNet 预训练 + 轻量分类器」上限，**「算力不够上 Swin-T/3B QLoRA」是伪问题，已有实测依据**。
泄漏幅度六格全部 <1pp。src：`exp/p1_d3_summary.json`。

**99.73% 的可信度已用三层递进验过**（每层判据必须**不可被前一层结论推出**）：
混淆矩阵（错 6/1912，recall 0.9936~1.0000，错误分散）→ 遮挡热区（跨类平均 r=**+0.015**，
有效块 3.2/16）→ **裁剪带对照（下三分之一 99.91%，比全图还高；上带 91.44%）**。
**结论：模型确实在读缺陷件，不是认机器本体/构图/拍摄距离。**
脚本：`exp/d3_error_analysis.py` / `d3_occlusion*.py` / `d3_where_look.py`。

⚠️ 照实记的局限：Stringing vs Layer_shifting 遮挡响应 **r=+0.890**（唯一高相关类对）
→ 两类「看的位置」没区分开，区分依据在特征整体方向上，单块遮挡测不出。
6 个错例全在 Cracking/Layer_shifting/Warping 三类互混，其中 **2 例人眼复核更认同模型** →
真实上限约 99.7~99.8%，但**报告仍报实测 99.69%，不擅自上调**。

⚠️ 这只排除了「认错东西」，**推不出「换机器/换材料也准」**——DC-02R 全来自同一台 Ender + 同批 PLA。
落地到别的机型**必须重采目标域数据**。唯一未闭合风险 = DC-04 外部验证卡网络（Zenodo 16 KB/s）。

## ✅ 映射层已看图定稿 5/5

`kb_mapping.py` 的 `pending_visual_check` 全清零。**改映射不需重训模型**（推理侧只用 classes 顺序）。

| 视觉类 | primary | secondary | confusable |
|---|---|---|---|
| Stringing | KB006, KB063 | — | KB101 |
| Warping | KB024, KB044, KB016 | — | — |
| Layer_shifting | KB008, KB050, KB052, KB065 | — | KB009 |
| **Cracking** | **KB009**（原 KB049） | KB049 | — |
| **Off_platform** | **KB101, KB102, KB103**（原 KB002/KB028） | KB002, KB028 | KB067 |

## 🔴 本项目最贵的两条教训

**① 字面对得上 ≠ 现象对得上。** 两条映射都是「按字面语义推断」定的，看着完全合理
（Cracking→裂纹、Off_platform→脱离平台），**看图后发现全错**：
- **Cracking**：20/20 是层向水平贯穿 + 撕裂纤维 + 层间错位 → 实质是**层间结合失效 Delamination**，
  主映射应是 **KB009**（不是"裂纹"字面对应的 KB049）。
- **Off_platform**：20/20 画面里**没有任何成形件**，材料全堆成乱丝 → 是 **Spaghetti 挤出失败**。
  而 kb_v2 里 **KB101 标题就是 `Spaghetti (Print Detached from Bed)`**，首条根因
  `First layer did not adhere properly` → **它才是主条目**，不能因"标题看起来像别的东西"降级为易混淆。
- ⚠️ 看图前必须知道：**DC-02R 构图 = 整台打印机 + 底下一小块件，件占比很小，只看缩略图会误判，必须放大。**

**② 判定「模型是否走了捷径」不能只看「该线索单独能到多少分」，必须主动移除线索看分数掉多少。**
实例：仅用 19 维低级像素统计就能拿 71.5% macro-F1、各类背景色两两分明 → 我据此**推断**
「99% 反映的是识别拍摄批次」→ **被干预实验推翻**：去背景 0.00pp、去色+去背景仅 −0.33pp。
低阶捷径客观存在但线性探针并不依赖它。脚本：`exp/audit_d3_credibility.py` + `exp/d3_intervention.py`。
延伸：遮挡指标**必须用连续值**（概率相对降幅），0/1 翻转在 5 分类里只有两档，
全是 0 只说明「没翻转」，不能说明「没信息」。

## ✅ D6 端到端 + 拒答

`src/vision/infer.py`（部署件落盘 `data/_model/` + 评估护栏）、`src/vision/pipeline.py`（图→5 类→KB+拒答）。
重建：`python -m src.vision.infer --train`（约 4.5 min）。

🔴 **铁律：部署件上的高置信度是自证，对新图毫无参考价值。**
实测：一张折外被误判的图（真值 Cracking / 折外判 Warping / 概率仅 0.6343），
在部署件上给出 **Cracking 0.9928**——因为它训练时见过这张图。
**① 部署件不能报 99.73%**（它是全量重训的，算 macro-F1 = 训练集准确率）。
`infer.evaluate()` **直接抛错拒绝执行**——与 P0「模型选择误用测试折」同类错误防复发。
**② 阈值必须用折外模型标定**：正确(n=1906) mean 0.9957/min 0.5576，错误(n=6) mean 0.6910/min 0.4919
→ **拒答阈值 0.90**（回答 98.6%、拦下错误 83.3%；曾拍脑袋定 0.60 只拦 33% 且会误伤）。
验收：4 个低置信度错例**折外模型拒答 4/4 触发**（`exp/d6_e2e_reject_check.py`）。
⚠️ 高置信度≠正确（0.90 下仍有 1/6 错误溜过）；只有 6 个错误样本，阈值只是起点。

## ✅ P0 ↔ P1 接线已打通（2026-10-05）

`src/skill_vision_diagnose.py` + `src/agent_vision.py`。**P0 冻结区一行未改**（git status 可验）。
思路：**让 P1 的输出长成 P0 的 evidence**——生成器只认 `kb_id/problem/text`，
与 `data/kb_chunks.json` 结构一致，故零改动。

🔴 **唯一真问题 = 上下文预算**：P0 的 `MAX_CONTEXT_CHARS=10000` 硬编码不可改，
视觉证据 7.2k~28.3k。**贪心截断会挤掉整块**（实测 Layer_shifting 只剩 KB008、
Off_platform 只剩 3/6，且模型不知道被丢了）。方案 = **max-min轮转填充**，5 类KB 全保留、
不触发二次截断，根因覆盖 54%（**10k 硬上限的诚实结果**：单条根因最大 3,453 字）。
⚠️ 每处裁剪都追加 `[…已截断 N 条根因]`，**绝不静默截断**。
❌ 「按剩余体量比例分配」**更差**：只花 6,768/10,000 且 KB008 只拿 1 条根因。

验收 `exp/p1_p0_wiring_check.py` → `exp/p1_wiring_result.json`，**A~F 六项全过**。

🔴 **抓到的 4 个 bug，①最危险**：
1. **LangGraph 静默丢弃 schema 未声明的入参**：`AgentState` 是 `TypedDict(total=False)`，
   未声明的 `image_paths` **直接消失、不报错** → 有图也走文本路径，返回值「看起来都正常」。
   **只能靠断言 `route=="vision"` 抓到**。修法见 `VisionAgentState`。
2. **多图合并所有 KB 给互斥建议**（3 张不同类别 → 12 KB / 3 套互斥建议）。
   **类别是互斥的诊断结论，不是可并集的信息**。改为只取主类别 + 显式告知冲突。
3. **验收断言写错**：`top_cosine` 两路径语义不同（文本=真实余弦 / 视觉=-1.0），
   写成恒等断言 → 文本路径必然 FAIL。**写断言前先问「这量在各路径语义相同吗」**。
4. **预算记账漂移**：固定 `HEADER_RESERVE` 估算 → 实际 10,146 > 10,000。改为逐字复算。

✅ **P0 既有 bug 已获批准并真修（2026-10-05，commit `181ca19`）**：
`load_fitted_embedder()` 硬编码读 cause 级索引路径，kb 模式读不到 fitted 模型
→ 用单条 query 重新 fit 出**迷你模型**。修法 =按路径取 + `_FITTED_EMBEDDER_CACHE`
**分桶** + `faiss_path` 透传 + `strict=True` 缺失即抛；验收旁路已换成
`assert_no_bypass_needed()` **反向自检**。A~F 重跑 ALL PASS。

🔴 **归因两次修正，别照抄旧结论**：原写「抛 `TruncatedSVD n_components` 越界」只对了一半。
- (a) 短 query（只切 1 个 token → `n_components=0`）→ **抛错**（吵闹，原判断对）
- (b) 长 query → **vocab=3** 的迷你模型查 16367 词表的索引，**维度同为 1024
  所以 FAISS 不报错**，静默返回假相似度（top1 KB002 cos=0.3613，正确是 KB006）
  → 与 `assert_backend_consistent` 同一类，**危害更大**，`strict=True` 是必要的。

⚠️ **修好后暴露一个独立既有缺陷（未修）**：离线兜底 TF-IDF 对**中文 query
双路皆零**——索引语料是英文 + 默认 `token_pattern` 不做中文分词 → `nnz=0`
→ 查询向量全零；BM25 被同一分词器拖累 → `retrieve("打印件表面有拉丝")`
返回 KB001~KB005 **字母序**。中文场景**必须配 `DASHSCOPE_API_KEY`**（生产本就如此）。
**不影响 P0 的 54.37%**（那是带 key 测的）。但这解释了「无 key 下 top_cosine 恒 0」。
明细与对照实验：`exp/_tmp_bug_mechanism.py` + `exp/p1_vision_result.md` §4.5.5。

## 环境（已就绪，别重装）

`torch 2.14.1+cu126 / tv 0.29.1+cu126 / timm 1.0.30`，CUDA 可用，RTX 4060 8GB，Python 3.13，
venv = `F:/workbuddy/3d-print-troubleshooting-agent/.venv`。`scripts/p1_check_env.py` 应 5/5 全绿。
⚠️ **必须 cu126**（cu124 索引下 cp313 只有 torch 2.6/tv 0.21）。装 CPU 版不报错但慢 20×。

## 数据（已定，别再讨论）

DC-02R = Kaggle `wengmhu/fdm-3d-printing-defect-dataset`，MIT，1,912 张（538/472/447/364/91），
尺寸 3072×2048，数据根 `D:/datasets/fdm-defect/FDM-3D-Printing-Defect-Dataset/data`（**两层嵌套**）。
**不需要 Kaggle 凭据**（公开，匿名可下）——别再让用户配。README 的 Cracking=427 是笔误，实为 472。
- **DC-04**（Zenodo 18628777, CC BY 4.0）= 外部验证集 + 正常件门控源，**不替代主集**
  （Zenodo 4 类缺 Warping/Cracking/Off_platform，换主集映射层就断）。只下 `test_data.zip` 12.6 GB。
- ⛔ 已排除：3D-ADAM（任务不匹配）；Garuda/JITK 2026 的 99.68%（数字高到不可信，不可作对标）。
- **DC-02R 无材料/参数/EXIF 维度**（文件名全是时间戳，唯一维度是 8 个拍摄日期）→
  **「按材料/参数分层验证」在此数据集上无法设计**，别再提。
- 去泄漏：前缀 GT（DC-02R 自带确定性，召回 100%）∪ DINOv2 余弦≥0.995 → **1,749 group**，
  产出 `data/p1_folds.json`。SSRN 7395570 验证 **pHash 仅召回 3.3% 真实同病灶组** → 不可只用 pHash。
  **group-aware 为主口径**，stratified 仅作对照算泄漏幅度。

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/vision/guards.py` | **13 道防线**，`--self-test` 全绿（骨干审计/目录探测/计数器/冒烟/反向自检/权重加载/AST 符号检查/pHash 降级/GT 召回/group 一组一折/泄漏度量合理性/iter_folds 必填/缺 fold 必 FAIL） |
| `src/vision/weights.py` | 健壮权重加载，**禁用 `timm.create_model(pretrained=True)`** |
| `src/vision/dedup.py` | 去泄漏 → 1,749 group + 两套折 |
| `src/vision/train.py` | 三级对照 L1/L2/L3，**读 D2 的折号不重做 KFold** |
| `src/vision/infer.py` / `pipeline.py` | 部署件落盘 + 端到端 + 拒答 |
| `src/vision/kb_mapping.py` | 映射层唯一权威源，导入即校验 KB ID 存在 |
| `src/skill_diagnose.py` / `src/agent_graph.py` | P0 Skill 契约 + LangGraph 编排（`generate` 只吃 `evidence`） |
| `src/skill_vision_diagnose.py` / `src/agent_vision.py` | **P0↔P1 接线**（视觉 Skill + 双路编排，含上下文预算水填充） |
| `exp/p1_p0_wiring_check.py` | 接线验收 A~F 六项 + `assert_no_bypass_needed()` 反向自检 |

## 踩过的坑（别重犯）

**环境**
- ⚠️ 本机 venv 缺 P0 依赖：`faiss-cpu`/`rank_bm25`/`langgraph` 已补装。
  P0 索引（`data/faiss_index_kb.*`，gitignore）**原本不存在**，需先跑
  `python src/build_kb_index.py`；无key 时走 TF-IDF+SVD 离线兜底。
- ⚠️ 机器有代理 `127.0.0.1:60318`；Zenodo 需 `curl --resolve zenodo.org:443:<真实IP>`
  绕 DNS 污染但仅 16 KB/s → DC-04 暂缓、换网络。pip 走同一代理可通 PyPI。
- ⚠️ git 身份未配置：用仓库既有身份 `avoid137 <1303572692@qq.com>`（不动全局配置）。

**权重加载**（`weights.py`，别绕过）
- 🔴 **禁用 `timm.create_model(pretrained=True)`**（guards/train/dedup 三处已换 `weights.build_model`）。
  `load_pretrained_into` 校验注入张量数，0 个即抛 `WeightError`（防静默退化成随机初始化）。
- `_read_state_dict` 用**魔数嗅探**而非后缀（HF blob 是 sha256 无扩展名但内容是 safetensors，
  按 `.pth` 走 torch.load 报 `UnpicklingError: Unsupported operand 176`）。
- `header too small` ≠ 网络失败（实测 blob 完整、snapshot 0 B，是 Windows 符号链接失败）。
  `EXPECTED_BYTES` 全为实测值：B0=21,355,344、DINOv2=**346,334,872**（按估计写会反复重下）。
- 唯一可用权重源 = **`hf-mirror.com`**（1.0 MB/s）；`huggingface.co` 超时、
  `modelscope.cn` status=000（不必再试）、`mirrors.aliyun.com` 不通；torch.hub 的 DINOv2 同样不可用。

**数据**
- 🔴 **pHash/dHash 已彻底降级为只读诊断、不参与 union**。误合并率 **94.3%**，
  传递闭包产出 **19 个跨类污染 group**。别回退。
- 🔴 **Kaggle `totalBytes` 元数据不可作完整性基准**：元数据 6,293,272,351 B vs 实际对象
  6,279,686,097 B，差 13.6 MB，是 Kaggle 侧元数据误差，按元数据补下会被拒 `InvalidRange`。
  **正确判据 = zip 条目数 1,912 + CRC 全过 + 5 类张数一致。**
- ✅ **不需要 Kaggle 凭据**（公开数据集，匿名可下）。**别再让用户配凭据。**
- ⚠️ `data/folds.json` 里 path 是**绝对路径**（数据集在 D 盘），做 `relative_to(ROOT)` 会 ValueError。
- ⚠️ **`data/mapping.md` 已失效，不可作映射层依据**（基于弃用的 8 类体系，8 个 KB ID 不在 kb_v2 里）。

**代码/工具**
- ⚠️ **别用 `except: pass` 掩盖导入失败**：`scripts/` 下 `sys.path[0]` 是 `scripts/` → 导入失败被吞 →
  下探静默失效、依然报 0 张，白排查半小时。用 `importlib.util.spec_from_file_location`
  按路径加载 + 显式 `sys.modules` 注册。
- ⚠️ **别凭记忆写 `old_string`**（改长段反复 3 次失败）——先 `sed -n` 看确切内容。
- ⚠️ **别用 `exp/*.png` 全局 ignore**（会挡掉报告配图）；只忽略调试中间产物。
- ⚠️ **长任务必须 `run_in_background` + `python -u`**（否则日志缓冲到结束才落盘）。

## 回迁

读 `HANDOFF_P1.md` → `python scripts/p1_check_env.py`（应 5/5 全绿）→ 读本文件 → 读 `exp/p1_vision_result.md`。
