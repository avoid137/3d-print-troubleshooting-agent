# 3D 打印故障诊断 Agent · 项目记忆（仓库自动加载）

> 本文件随仓库 clone 到新机器；把仓库根目录作为 WorkBuddy 工作区打开时自动注入。
> 完整接手文档始终是仓库根的 **HANDOFF_P1.md**（权威、面向 Agent 与人都可读）。
> 详细实测数字在 `exp/p1_vision_result.md`（本文件只留「不能忘的」）。

## 当前主线：P1（视觉分类，RTX 4060 机器）

- **P0 已交付且严格验证到顶**（别再碰）：strict SOTA **54.37%**(87/160, qwen-turbo 判官，
  判官间一致率 74% → 不确定度约 **±7pp**)、lenient 69.37%、拒答误拒 0。
  检索侧（top-k=10/rerank/query 改写/索引变体 8 个）与生成层（v2.3 净负、v2.8 +2.5pp 但 23 回归）
  全部证伪或不过零回归安全门。结论见 `exp/generation_layer_campaign_conclusion.md`。
- **P1 任务**：5 类缺陷图像分类（Kaggle `wengmhu`/DC-02R，MIT，1,912 张）+ 视觉→知识库映射层。
- **P1 只动 `src/vision/`**，不要改 P0 检索/生成层（重做必回归）。
- **当前唯一未闭合风险**：DC-04 外部验证卡在网络（Zenodo 16 KB/s，78 小时）。
  §2.4.3 已证明「模型读的是缺陷件」，**但推不出「换机器/换材料也准」**——
  DC-02R 全来自同一台 Ender + 同批 PLA。落地到别的机型**必须重采目标域数据**。
- **P0 ↔ P1 尚未接线**：P0 只吃文本、不接受图片。接线需动 P0 → 需用户确认。

## ✅ P1 核心结论（2026-10-04 实测定稿，数字勿改）

**选定方案 = L2 冻结骨干 + 线性探针，零训练成本。**

| 级别 | EfficientNet-B0 | MobileNetV3-small |
|---|---|---|
| L1 冻结+kNN | 98.44 ± 0.57% | 98.16 ± 0.80% |
| **L2 冻结+线性探针** | **99.73 ± 0.06%** | **99.41 ± 0.70%** |
| L3 全量微调 | 99.19 ± 0.98% | 99.25 ± 0.48% |

（group-aware 折，5 折 × 2 种子 = 10 次/格，src：`exp/p1_d3_summary.json`）

- **L3 无增益**（−0.54pp / −0.16pp，两骨干复现），B0 上 σ 是 L2 的 16 倍。**别再跑 L3。**
- **两骨干完全等价**（差 0.06~0.31pp）→ 已触及「ImageNet 预训练 + 轻量分类器」上限。
  **「算力不够所以上 Swin-T / 3B QLoRA」这个伪问题已有实测依据。**
- 泄漏幅度六格全部 <1pp → 可忽略（折叠 163/1912=8.5%；普通分层实际跨折 234 张=12.24%）。
- 🔴 **对外口径**：99.73% 只能说「DC-02R 内、group-aware 折、5 类 macro-F1」，
  **不能**说「缺陷识别准确率 99.7%」或「SOTA」。

## ✅ 99.73% 已通过三层可信度递进（2026-10-04）

| 层 | 判据 | 实测 | 排除了什么 |
|---|---|---|---|
| 1 混淆矩阵 | 是否系统性退化 | 错 6/1912（0.31%），recall 0.9936~1.0000，错误分散 4 个类对 | 「整体把 A 当 B」 |
| 2 遮挡敏感度 | 模型盯图哪一块 | 跨类热区平均 r=**+0.015**，有效块 3.2/16 | 「靠共性区域一锅端」 |
| 3 **裁剪带对照** | 去掉其余还剩多少 | **下三分之一 99.91%（比全图还高）**，上带 91.44% | 「认机器本体/构图/拍摄距离」 |

**结论：模型确实在读缺陷件。** 全图 99.73% 不是虚高。
脚本：`exp/d3_error_analysis.py` / `d3_occlusion.py` / `d3_occlusion_similarity.py` / `d3_where_look.py`。

⚠️ 已知局限（照实记，别藏）：
- Stringing vs Layer_shifting 遮挡响应 **r=+0.890**（唯一高相关类对，峰值同为 (2,2)）
  → 这两类「看的位置」没区分开，区分依据在特征整体方向上，单块遮挡测不出。
- 6 个错例全是 Cracking/Layer_shifting/Warping 三类互混，其中 **2 例人眼复核更认同模型而非标签**
  → 真实上限约 99.7~99.8%，但无人工复核集无法证明确切值，**报告仍报实测 99.69%，不擅自上调**。

## ✅ 映射层已看图定稿 5/5（2026-10-05，**两条推断都被推翻**）

`kb_mapping.py` 的 `pending_visual_check` 已全清零。**改映射不用重训模型**（推理侧只用 classes 顺序）。

| 视觉类 | primary | secondary | confusable |
|---|---|---|---|
| Stringing | KB006, KB063 | — | KB101 |
| Warping | KB024, KB044, KB016 | — | — |
| Layer_shifting | KB008, KB050, KB052, KB065 | — | KB009 |
| **Cracking** | **KB009**（原 KB049，已改） | KB049 | — |
| **Off_platform** | **KB101, KB102, KB103**（原 KB002/KB028，已改） | KB002, KB028 | KB067 |

🔴 **两条被纠正的推断**（看 20 张图得出，与原方向相反）：
- **Cracking**：原写「主 KB049（裂纹）、KB009 为易混淆」。实际 20/20 是**层向水平贯穿裂纹
  + 撕裂纤维 + 层间错位** → 实质是层间结合失效（Delamination），**主映射应为 KB009**。
- **Off_platform**：原写「主 KB002/KB028（不粘床）、KB101 归 confusable（标题相近机理不同）」。
  实际 20/20 画面里**没有任何成形件**，材料全堆成乱丝 → 是**Spaghetti 挤出失败**。
  🔴 而 `kb_v2` 里 **KB101 的标题就是 `Spaghetti (Print Detached from Bed)`**，
  首条根因 `First layer did not adhere properly`，**它才是本类的主条目**。

> 🔴🔴 **本项目最贵的一条教训**：两条映射都是「按字面语义推断」定的，
> 看着合理（Cracking→裂纹、Off_platform→脱离平台，字面对得上），**看图后发现全错**。
> **字面对得上 ≠ 现象对得上。** KB 条目标题可能包含**旧版的、正当的别名**
> （KB101 就叫 "Print Detached from Bed"），**不能因为「标题看起来像别的东西」就降级为易混淆项**。
> ⚠️ 看图前必须先知道：**DC-02R 构图 = 整台打印机 + 底下一小块件，件占比很小，
> 只看缩略图会误判，必须放大。**

## ✅ D6 端到端已打通（2026-10-05）

`src/vision/infer.py`（部署件落盘 + 评估护栏）+ `src/vision/pipeline.py`（图像→5 类→KB + 拒答）。
落盘 `data/_model/`（已 gitignore），`python -m src.vision.infer --train` 约 4.5 min。

🔴 **两条铁律**：
1. **部署件不能报 99.73%**。99.73% 来自 group-aware 5 折**折外**预测；
   部署件是**全量 1,912 张重训**的，在它上面算 macro-F1 = 训练集准确率，必然虚高。
   `infer.evaluate()` **直接抛错拒绝执行**。与 P0「模型选择误用测试折」同类。
2. **部署件上的高置信度是自证，对新图毫无参考价值**。实测：一张折外被误判的图
   （真值 Cracking / 折外判 Warping / 概率仅 0.6343），在部署件上给出 **Cracking 0.9928**
   ——因为它训练时见过这张图。**标定阈值必须用折外模型。**

**拒答阈值 0.90（实测标定，非拍脑袋）**：
```
正确(n=1906) 最高概率 mean 0.9957 / min 0.5576
错误(n=  6) 最高概率 mean 0.6910 / min 0.4919  → 置信度能区分对错
th=0.90 → 回答 98.6% 拦下错误 83.3%（0.60 只会拦 33% 且会误伤）
```
⚠️ 高置信度≠正确（0.90 下仍有 1/6 错误溜过）；⚠️ 只有 6 个错误样本，阈值只是起点。
验收：报告 §五 的 4 个低置信度错例，**折外模型拒答 4/4 触发**（`exp/d6_e2e_reject_check.py`）。

## 🔴 本项目最重要的两条方法论纪律

**① 判定「模型是否走了捷径」不能只看「该线索单独能到多少分」，
必须主动移除该线索、看分数掉多少。**

实例：DC-02R 仅用 19 维低级像素统计（无空间结构）就能拿 71.5% macro-F1，
各类背景色两两分明 → 我据此**推断**「99% 反映的是识别拍摄批次而非缺陷形态」。
**该推断被干预实验推翻**：去背景 0.00pp、去色+去背景仅 −0.33pp。
低阶捷径客观存在但线性探针并不依赖它。脚本：`exp/audit_d3_credibility.py` + `exp/d3_intervention.py`。

**延伸**：高分可信度**不能只做一步**，且每层判据必须**不可被前一层结论推出**——
混淆矩阵（是否系统性）→ 遮挡（看哪块）→ 裁剪带（去掉其余还剩多少），三层全通过才敢下结论。
另：遮挡指标**必须用连续值**（概率相对降幅），0/1 翻转在 5 分类里只有两档，
全是 0 只说明「没翻转」，不能说明「没信息」。

**② 映射必须先看图再定，禁止按字面语义推断**（见上面映射层那节的教训）。

## 环境（已就绪，**别再重装**）

```
torch 2.14.1+cu126 / torchvision 0.29.1+cu126 / timm 1.0.30
CUDA 可用: True · NVIDIA GeForce RTX 4060 · 8.0 GB · CUDA 运行时 12.6
venv = F:/workbuddy/3d-print-troubleshooting-agent/.venv
p1_check_env.py → 5/5 全绿
```
⚠️ **必须用 cu126**（非 cu124）：本机 Python 3.13，cu124 索引下 cp313 只有 torch 2.6/tv 0.21。
4060 需 CUDA 版，装 CPU 版不报错但慢 20×。

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/vision/guards.py` | **13 道防线**（骨干审计/目录探测/运行时计数器/冒烟/反向自检/权重加载/AST 符号检查/pHash 降级/GT 召回 100%/group 一 group 一折/泄漏度量合理性/iter_folds 必填/缺 fold 必 FAIL），`--self-test` 全绿 |
| `src/vision/weights.py` | 健壮权重加载，**禁用 `timm.create_model(pretrained=True)`**（见下） |
| `src/vision/dedup.py` | 去泄漏：前缀 GT（DC-02R 自带确定性）∪ DINOv2 余弦≥0.995 → 1,749 group，产出 `data/p1_folds.json`（两套折） |
| `src/vision/train.py` | 三级对照 L1/L2/L3，**读 D2 的折号不重做 KFold**；L3 早停用训练折内 12% 验证集 |
| `src/vision/infer.py` | 部署件落盘 + 评估护栏 |
| `src/vision/pipeline.py` | 端到端 + 拒答 |
| `src/vision/kb_mapping.py` | 映射层唯一权威源，导入即校验 KB ID 存在 |
| `exp/p1_vision_result.md` | **完整结果报告**（所有实测数字） |

## 踩过的坑（别重犯）

- 🔴 **pHash/dHash 已彻底降级为只读诊断、不参与 union**。实测误合并率 **94.3%**
  （2,539 对里 2,376 对是误合并），且传递闭包产出 **19 个跨类污染 group**。**别回退。**
  详见 `src/vision/dedup.py` docstring。
- 🔴 **权重加载**（`weights.py`，别绕过）：
  - `header too small` ≠ 网络失败。实测 blob = 21,355,344 B **完整**，snapshot = **0 B**（符号链接失败）。
  - **禁止 `timm.create_model(..., pretrained=True)`**（guards/train/dedup 三处已换成 `weights.build_model`）。
  - `load_pretrained_into` 校验注入张量数，注入 0 个即抛 `WeightError`——防静默退化成随机初始化。
  - `_read_state_dict` 用**魔数嗅探**而非后缀（HF blob 是 sha256 无扩展名，内容却是 safetensors；
    按 `.pth` 走 torch.load 会报 `UnpicklingError: Unsupported operand 176`）。
  - `EXPECTED_BYTES` **全为实测值**：B0 = 21,355,344；DINOv2 = **346,334,872**
    （最初按估计写 346,584,396，差 249 KB，会把完好文件误判为截断而反复重下）。
  - 唯一可用权重源 = **`hf-mirror.com`**（1.0 MB/s）。`huggingface.co` 超时、
    `modelscope.cn` status=000（不必再试）、`mirrors.aliyun.com` 不通。torch.hub 的 DINOv2 同样不可用。
- ✅ **不需要 Kaggle 凭据**。DC-02R 是**公开**数据集，匿名即可下载。**别再让用户配凭据。**
- 🔴 **Kaggle `totalBytes` 元数据不可作为完整性基准**：元数据报 6,293,272,351 B，
  实际对象 6,279,686,097 B，差 13.6 MB。**这是 Kaggle 侧元数据误差，不是下载缺失**，
  按元数据补下会被拒为 `InvalidRange`。
  **正确判据 = zip 条目数 1,912 + CRC 全过 + 5 类张数一致。**
- ⚠️ **别用 `except: pass` 掩盖导入失败**。`scripts/` 下 `sys.path[0]` 是 `scripts/` → 导入失败被吞
  → 下探静默失效、依然报 0 张，白排查半小时。正确做法：`importlib.util.spec_from_file_location`
  按路径加载 + 显式 `sys.modules` 注册（dataclass 解析需要）。
- ⚠️ **别用 `exp/*.png` 全局 ignore**（会把报告配图一起挡掉）；只忽略调试中间产物。
  `data/_model/`、`data/_weights/`、`exp/_cache/`、`exp/_*.json`、`exp/d3_*.log` 需 gitignore。
- ⚠️ **DC-02R 的 `folds.json` 里 path 是绝对路径**（数据集在 `D:/datasets`），
  对它做 `relative_to(ROOT)` 会 ValueError。
- ⚠️ **长任务必须 `run_in_background` + `python -u`**（不加 `-u` 日志缓冲到结束才落盘，实时看不到进度）。
  裁剪带 5 档 × 1912 张前向 + 25 次 LogReg ≈ **18 分钟**。
- ⚠️ **机器上有 HTTP 代理 `127.0.0.1:60318`**，Zenodo 需 `curl --resolve zenodo.org:443:<真实IP>`
  绕 DNS 污染，但仅 16 KB/s → **DC-04 暂缓，换网络环境**。pip 走同一代理可通 PyPI。
- ⚠️ **git 身份未配置**：用仓库既有身份 `avoid137 <1303572692@qq.com>`（不动全局配置）。
- ⚠️ **`data/mapping.md` 已失效，不可作映射层依据**（基于弃用的 8 类体系，8 个 KB ID 不在 kb_v2 里）。

## 数据集定案（已定，别再讨论）

- **DC-02R 主集**：Kaggle `wengmhu/fdm-3d-printing-defect-dataset`，MIT，1,912 张。
  数据根 = `D:/datasets/fdm-defect/FDM-3D-Printing-Defect-Dataset/data`（**两层嵌套**，别只 iterdir 一层）。
  张数 538/472/447/364/91，尺寸统一 3072×2048。
  ⚠️ Kaggle README 的 Cracking=427 是笔误，实为 472。
- **去泄漏方案**：前缀（`_aug`/`_original` 去后缀，DC-02R 自带确定性 GT，召回 100%）
  ∪ DINOv2 余弦≥0.995。SSRN 7395570 验证 **pHash 仅召回 3.3% 真实同病灶组** → 不可只用 pHash。
  **group-aware 为主口径**，stratified 仅作对照算泄漏幅度。产出 `data/p1_folds.json`。
- **DC-04（Zenodo 18628777，CC BY 4.0）** 定位为**外部验证集 + 正常件门控源**，不替代主集
  （Zenodo 4 类无 Warping/Cracking/Off_platform，**换了主集映射层就断**）。只下 `test_data.zip`（12.6 GB）。
- ⛔ **已排除**：3D-ADAM（NeurIPS 2025，RGB+3D 点云，任务不匹配）；
  Garuda/JITK 2026 的 99.68%（数字高到不可信，**不可作对标**）。
- 验收：macro-F1 ≥ 0.80 + **必须报跨数据集差值** + 5 折 × 2 种子。
- 💡 **算力不是瓶颈**：5 折 × 2 套划分 = 10 次训练，4060 全程约 20~30 分钟。

## 纪律红线

1. **数字做完才填**，禁止预估。
2. **P1 不改 P0 检索/生成层**（重做必回归）。
3. **原始数据不进 git**；权重/模型/缓存全部 gitignore。
4. **许可明确才用**（DC-02R MIT、DC-04 CC BY 4.0 需署名）。
5. **映射不硬凑**，必须先看图（见纪律②）。
6. **对外口径**：99.73% 只说「DC-02R 内、group-aware 折、5 类 macro-F1」。

## 回迁

接手新机器：读 `HANDOFF_P1.md` → 跑 `python scripts/p1_check_env.py`（应 5/5 全绿）
→ 读本文件 → 读 `exp/p1_vision_result.md`。
