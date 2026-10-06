# P1 接力包 · 转移到 4060 机器

> 生成时间：2026-10-04，**最近更新：2026-10-06** ｜ 源机器：本机（D:/3D打印故障诊断Agent）
> 任务：**P1 — 5 类缺陷图像分类 + 视觉→知识库映射层**
> 一句话：P0 检索层已交付并可部署，剩下的活需要 GPU；这套包让你在 4060 机器上直接开工，不用回头翻聊天记录。

---

## 1. 一页速览

| 项 | 状态 | 数字（实测，来源可溯） |
|---|---|---|
| **P0 检索层** | ✅ 完成 | Recall@5 = **90.00%**（144/160）；gold⊆top5 = 80.63%（129/160） |
| **P0 拒答层** | ✅ 完成 | 拒答 F1 **1.0000**；in-scope 误拒 **0/160** |
| **P0 端到端 SOTA** | ✅ 可部署 | strict **54.37%**（87/160）／ lenient **69.37%**（111/160），零回归 |
| **P1 视觉分类** | ✅ 完成 | **DC-02R 内 group-aware 折外macro-F1 = 99.73%**（5 类 / EffNet-B0 冻结+线性探针） |
| **P1 映射层** | ✅ 完成 | 5/5 **看图定稿**，改映射不需重训 |
| **P1 端到端+拒答** | ✅ 完成 | 拒答阈值 **0.90**（折外标定），DC-02R 200/200 |
| **P0↔P1 接线** | ✅ 完成 | 验收 A~F 全过（`exp/p1_p0_wiring_check.py`） |
| **外部验证（D8）** | 🔴 **负结果** | DC-04 上 6.25%/8.25%（低于随机 20%）→ **模型无域适应能力** |
| **正常类（D9）** | ⚠️ **代码就绪但默认关闭** | DC-04 正常件会吞掉缺陷域（缺陷件 100% 误判正常）→ `NORMAL_TRAIN=0` |

**P1 目标（README 待办第 328 行原文）**：接入 5 类缺陷图像分类（Kaggle `wengmhu`，MIT，1,912 张），构建视觉 → 知识库映射层。

### 🔴 对外口径（别越线）

只能说「**DC-02R 内、group-aware 折、5 类 macro-F1 99.73%**」。
**不能说**「缺陷识别准确率 99.7%」或「SOTA」——
DC-02R 全来自同一台Ender + 同批 PLA，换机器/材料**未验证且已知会失败**（见 §10）。

---

## 2. 迁移三步走（按顺序做，每步有验收）

### 步骤 A：代码（推荐直接 clone，别手工拷）

```bash
git clone https://github.com/avoid137/3d-print-troubleshooting-agent.git
cd 3d-print-troubleshooting-agent
```

- 远端已含本接力包（HEAD = `1b149af`，2026-10-04 push）。
- private 仓，新机器需先配置 GitHub 凭据（PAT 或本机凭据管理器里的 token）。
- ⚠️ 若新机器 push 报证书错误（SteamTools/Watt 加速器做 HTTPS 中间人）：
  `git -c credential.helper= -c credential.helper=wincred -c http.sslBackend=openssl -c http.sslCAInfo="<combined-ca-bundle.pem路径>" push origin main`
- 若手工拷贝：**不要**拷 `data/chroma_db/`、`*.pkl`、`*.bin`（大且可重建）。

### 步骤 B：环境（4060 专用）

```bash
# 1) 建虚拟环境
python -m venv .venv && .venv/Scripts/activate      # Windows

# 2) PyTorch（必须走官方 CUDA 索引，否则装成 CPU 版 → 训练慢 20 倍且不报错）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   必须打印 True + NVIDIA GeForce RTX 4060

# 3) 其余依赖
pip install -r requirements-p1-vision.txt

# 4) 自检（全绿才往下）
python scripts/p1_check_env.py
```

### 步骤 C：数据（6.3 GB，落 D 盘，别放 C 盘）

```bash
# 凭据三选一（密钥不要进仓库、不要贴对话）：
#   ① ~/.kaggle/kaggle.json = {"username":"...","key":"..."}
#   ② 环境变量 KAGGLE_USERNAME + KAGGLE_ACCESS_TOKEN
#   ③ 从本机拷贝 ~/.kaggle/access_token 到新机器同路径
python scripts/p1_fetch_dataset.py --dir D:/datasets/fdm-defect
```

脚本自带**完整性校验**，必须与 DC-02R 实测基准一致：

| 类别 | 期望张数 | 对应知识库条目 | 映射难度 |
|---|---|---|---|
| Warping | 538 | `Warping` | ✅ 同名 |
| Cracking | 472 | 层分离/结构类 | ⚠️ 需逐条核对 |
| Stringing | 447 | KB006 `Stringing or Oozing` | ✅ 同名 |
| Layer_shifting | 364 | `Layer Shifting` | ✅ 同名 |
| Off_platform | 91 | 附着类（首层不粘床等） | ⚠️ 需逐条核对 |
| **合计** | **1,912** | | |

> ⚠️ Kaggle 上游 README 写 `Cracking=427` 是**笔误**（数字颠倒），实测 **472**；以文件清单为准（538+472+447+364+91=1,912 自洽）。别拿 README 数字对不上就怀疑下载坏了。

验收：`data/p1_dataset_manifest.json` 生成且 `ok: true`。

---

## 3. P1 训练方案（我的倾向 + 理由）

**数据集只有 1,912 张、5 类**，这是小样本迁移学习场景，不是大数据集炼丹。所以：

| 决策 | 我的建议 | 理由 |
|---|---|---|
| 骨干 | **EfficientNet-B0 / MobileNetV3**（timm，ImageNet 预训练） | 小数据集上大模型必过拟合；且项目定位是"轻量可部署"，与论文 *Real-time defect detection via lightweight model deployment* 同调 |
| 输入尺寸 | 224×224（先跑通）；若指标不够再试 288/384 | 缺陷是纹理/几何形态，224 通常够 |
| 划分 | **分层 5 折（stratified 5-fold）**，固定 `seed=42` | Off_platform 仅 91 张，单折测试集才 ~18 张，单次划分指标噪声极大 |
| 类别不平衡 | 类别权重 `class_weight='balanced'` 或加权采样 | 538 : 91 ≈ 6:1，不处理会把小类吃掉 |
| 主指标 | **macro-F1（5 折均值 ± 标准差）** + 混淆矩阵 | accuracy 会被大类主导；选题就是缺陷识别，小类召回是业务价值 |
| 增强 | 水平翻转 + 轻色抖 + 随机裁剪；**不要**强几何扭曲 | 打印件有方向性，强扭曲会造伪样本 |
| 训练量 | 每折 ~20 epoch，早停 patience 5 | 1912 张 @ 4060 几分钟/epoch |
| 对标 | 论文 Swin-T **F1 87% / Acc 87%**（DC-02 公开数字，8 类任务，非同数据集） | 只能作**参照系**，不能直接比；写报告时要注明"任务与数据集不同，不可直接比较" |

**建议的目录与产物**（避免事后找不到证据）：
```
src/vision/train.py            # 训练（支持 --fold/--model/--img/--seed）
src/vision/dataset.py          # 数据读取 + 分层划分
src/vision/eval_map.py         # 视觉类 → 知识库条目映射
exp/p1_vision_result.md        # 结论报告（含混淆矩阵、失败案例）
data/p1_dataset_manifest.json  # 数据根与实测计数（脚本已生成）
```

---

## 4. 防坑清单（开训前必读）

**先加载本机 skill：`kaggle-vision-pipeline-guards`**（路径 `~/.workbuddy/skills/kaggle-vision-pipeline-guards/`；新机器若没装 WorkBuddy，就手工拷该目录过去）。

核心是"**不报错的错误最贵**"——loss 正常下降、分数就是不高，最可能是数据根本没喂进去。五道防线：

1. **骨干审计**：打印实际加载的骨干名/参数量/归一化 mean·std；归一化常数一律用 `timm.resolve_data_config()`，**绝不手写**（换骨干不换 mean/std 会静默劣化）。
2. **目录层级探测**：本数据集是 `data/<类别>/*.jpg`，脚本已自动下探；但**开训前抽样报告每类命中率**，< 0.5 就硬失败。
3. **运行时计数器**：累计空样本/读取失败数；空样本 > 5% → 先查管线别改模型。
4. **无数据冒烟测试**：`--smoke-test` 用合成数据跑通全链路（64px、1 epoch、CPU 可跑），抓形状 bug 成本几乎为零。
5. **防线自检**：写完任何断言，先验证它"条件不满足时确实会 fail"——永远通过的检查是装饰。
6. **权重加载防线**（2026-10-04 新增，见 `src/vision/weights.py`）：加载完预训练权重要**校验注入张量数**，注入 0 个就抛错——防「以为用了预训练其实随机初始化」。

> ⚠️ **2026-10-04 实测推翻了两条旧结论，务必先看**（RTX 4060 / Windows / Python 3.13）：
>
> 1. **DC-02R 不需要 Kaggle 凭据**。该数据集是**公开**的，匿名即可下载：
>    `kaggle datasets download wengmhu/fdm-3d-printing-defect-dataset` 无凭据直接开始传 5.85 GB。
>    不要再让用户配 `~/.kaggle/kaggle.json`（旧版本文档写"必须配置"是错的）。
> 2. **`SafetensorError: header too small` 不是网络问题**，是 Windows 专属的两个坑叠加：
>    - `huggingface_hub` 在**未开开发者模式的 Windows** 上无法创建符号链接，静默留下 **0 字节** snapshot，而 blob 本身完整（实测 blob 21,355,344 B 完好 / snapshot 0 B）；
>    - `hf_hub_download`（含 `local_dir=`）的传输会被**截断成 0 字节**，而同一 URL 用 `curl` 完整下载（@1.08 MB/s）。
>
>    对策已固化为 `src/vision/weights.py`：**禁止 `timm.create_model(..., pretrained=True)`**，
>    改用 `weights.build_model()`（四级查找 + 注入张量数校验 + 魔数嗅探读 safetensors）。
>    `guards.py` / `train.py` / `dedup.py` 三处均已替换。已实测注入：EfficientNet-B0 358 张量、DINOv2 174 张量（0 形状不符）。
>
> **可用镜像（关加速器 + `unset *_PROXY` 后实测）**：
> `hf-mirror.com` 200 @1.0 MB/s ✅ ｜ `kaggle.com` 200 ✅ ｜ `huggingface.co` 超时 ❌ ｜
> **`modelscope.cn` 魔搭 status=000 不可达 ❌（不必再试）** ｜ `mirrors.aliyun.com` ❌
>
> **Kaggle 并发下载**：1 线程 ~250 KB/s → **8 线程 ~697 KB/s**（5.85 GB ≈ 2.4 h）→ 16 线程无增益。
> CDN 支持 HTTP Range，签名 URL 有效期 3 天，断了重取 URL 续传。

**本项目特有的三个坑**：
- **Off_platform 只有 91 张**：5 折里每折测试约 18 张，单折 F1 抖动可达 ±10pp，**必须报 5 折均值±标准差**，不能只报某一折。
- **数据不对外分发**：只提交代码与索引脚本，原始图不进 git（`.gitignore` 要加 `data/raw/`、`data/datasets/`）。
- **类别映射别硬凑**：3 类同名可直接映射；Cracking / Off_platform 要**逐条核对知识库**后再定，写报告时标注"人工核对依据"。

---

## 5. 验收标准（做完才算 P1 完成）

| 验收项 | 门槛 | 备注 |
|---|---|---|
| 5 折 macro-F1 | **≥ 0.80**（对标论文 87%，但数据集/类别数不同，不可直接比） | 报均值 **± 标准差** |
| 各类别 recall | 最弱类（Off_platform）**≥ 0.70** | 小类不能塌 |
| 映射层 | 5 类 → 知识库条目映射表 100% 覆盖，含人工核对依据 | 存 `src/vision/eval_map.py` 或 `data/mapping.md` |
| 失败案例 | ≥ 5 例，带图 + 误判去向 | PMF 材料要用 |
| 复现性 | `seed` 固定、划分落盘（fold 分配存 json） | 否则换机器就对不上 |

⚠️ **诚信红线**（沿用 P0 纪律）：数字做完才填；未达到就写"持续迭代中"；不把论文 87% 说成自己的结果。

---

## 6. 遗留线索：P0 还能改什么（回来再做）

我在转 P1 前做了一轮改进空间分析，结论按性价比排序：

### 6.1 🔴 最重大发现：端到端指标有 ±7pp 的测量噪声

用 **qwen-plus** 做独立判官重判同一批生产答案（partial 样本 n=96，全量 160 未完成）：

| 判官 | correct 率 | 一致率 |
|---|---|---|
| qwen-turbo（现役） | 53.12% | **73.96%** |
| qwen-plus | 60.42% | |

**含义**：两个判官在 26% 的题上判得不一样，同一批答案换判官就差 **+7.29pp**。
→ 任何 **< 7pp 的端到端"增益"都淹没在判官噪声里**，不可作为结论。连回填的 +6.25pp 都处于噪声边缘。

**建议**：
1. 对外口径必须标注"strict 54.37%（qwen-turbo 判官口径；判官间一致率 74%，测量不确定度约 ±7pp）"。
2. 回填的 +6.25pp 需做**双判官交叉验证**（turbo + plus 都涨才算数）。
3. 今后 A/B 的显著性门槛提到 **> 7pp**，否则只报"等价"。

未完成：`exp/exp_judge_audit.py`（已跑 96/160，续跑命令 `python exp/exp_judge_audit.py`，有断点续跑，缓存命中不重复花钱）。

### 6.2 缺口分解（160 题归因，互斥且自洽）

| 归因 | 题数 | 占比 | 归属层 |
|---|---|---|---|
| correct | 87 | 54.4% | — |
| **检索漏召（gold⊄top5）导致失败** | 29 | 18.1% | 检索层 |
| **检索命中但漏根因**（partial） | 26 | 16.2% | 生成层 |
| **检索命中但答错**（wrong） | 18 | 11.2% | 生成层 |
| 检索漏召但答对（模型凭知识） | 2 | 1.3% | — |

按类别：multi_defect 最弱（43.3%，其中漏召 10/30）。按 gold 条数：gold=1 条 59.0% vs gold=2 条 39.5%（**多根因题是硬骨头**）。

### 6.3 检索侧：**已证伪，别再试**

离线探针（`exp/exp_retrieval_headroom.py`，零 LLM 生成）测了 8 个变体：

| 变体 | cov@5 |
|---|---|
| **A0 基线 BM25(full)+Vec(full)** | **129 (80.63%)** ← 最好 |
| A1 problem 词面加权×3 | 129（打平） |
| A2/A3 纯症状向量 | 113~115（**更差**） |
| A4~A7 双向量 max / cause 短文本 | 118~128（均不及基线） |

**结论**："症状信号被 solution 长文本稀释"的假设**不成立**——反而证明 solution 文本提供了有效语义。检索文本构造这条路**整类排除**。
已证伪的还有：top-k=10（16 题进上下文但生成不买账，净 −1）、rerank（gold/噪声分数无区分度）、query 改写（−3.75pp）。

### 6.4 生成层：已到顶

v2.3（净负 −3.76pp）、v2.8（+2.5pp 但 23 道回归）、v2.8b（与 v2.8 等价）全部失败。
**统一死结**：让模型做"相关性排除"必丢 gold；唯一安全模式是 **append-only 回填**。详见 `exp/generation_layer_campaign_conclusion.md`。

### 6.5 还有哪些没做（按性价比）

| # | 事项 | 成本 | 预期 |
|---|---|---|---|
| 1 | **判官审计补完 + 双判官口径定版** | ~20min | 让所有数字可辩护（**最该做**） |
| 2 | 幻觉 kb_id 检测（答案引用了不在 KB 的 id） | 零成本离线扫描 | 确定性 bug，修了必涨 |
| 3 | 去掉 few-shot 里的 KB-Demo 残留泄漏 | 小改 | 预估 ~+4-5 题 |
| 4 | kb_v2 补出处（现 3/56） | 数据活 | 合规与可信度 |
| 5 | 第三方标注泛化验证 | 需外部人力 | 证"非自说自话" |

---

## 7. 纪律红线（P0 沿用，别破）

1. **不对外分发原始数据**；仓库只提交代码 + 索引脚本，README 指路来源。
2. **只采用许可明确的源**：DC-02R 是 **MIT**（可放心用）；许可 Unknown 的一律不入库。
3. 数据集卡片 `data/dataset_cards.md` 是**唯一数据源台账**，下载/使用新数据集必须登记（来源、许可、规模、下载日期）。
4. 数字做完才填；跨材料口径必须一致（同一事实在不同文档里数值必须相同）。
5. 涉及诚信的表述绝不含糊：未完成的写"持续迭代中"，不夸大架构（如"Meta-Harness 自进化"）。

---

## 8. 回迁：P1 成果怎么带回

在新机器同一仓库里开发，直接 push（P1 代码建议放 `src/vision/`，报告放 `exp/p1_vision_result.md`）。

| 产物 | 是否进 git |
|---|---|
| `src/vision/*.py`、`exp/p1_vision_result.md`、`data/p1_dataset_manifest.json` | ✅ 提交 |
| 训练权重 `*.pth` / `*.pt` | ❌ 不提交（或放 GitHub Release） |
| 原始图像 | ❌ 绝不提交 |
| 折划分 json（seed + fold 分配） | ✅ 提交（可复现性证据） |

回到本机：`git pull` 即可，两边保持一致。

---

## 9. 给新机器 WorkBuddy 的接手方案（双保险）

> 目标：让新电脑上的 WorkBuddy 一接手就有完整上下文、不踩已证伪的死路、不浪费 4060 算力。
> 两种接法任选，推荐**接法 A（粘贴提示词）**为主、**接法 B（自动加载记忆）**为辅。

### 接法 A：粘贴开场提示词（最稳，不依赖记忆自动注入）

新机器 clone 完、仓库当工作区打开后，把下面这段**原样发给 WorkBuddy** 即可：

```
你接手「3D 打印故障诊断 Agent」项目的 P1 子任务。请先完整阅读仓库根目录的 HANDOFF_P1.md（这是唯一权威接手文档），再读以下文件建立上下文：
- data/dataset_cards.md（数据集台账，唯一数据源）
- requirements-p1-vision.txt（视觉环境依赖）
- scripts/p1_fetch_dataset.py、scripts/p1_check_env.py（已就绪的下载/环境自检脚本）
- exp/generation_layer_campaign_conclusion.md（P0 生成层已到顶的结论，别再碰）

P1 任务：在 RTX 4060 上训练 5 类缺陷图像分类（Kaggle wengmhu / DC-02R，MIT 许可，1,912 张），并构建「视觉类 → 知识库条目」映射层。严格按 HANDOFF_P1.md 的「迁移三步走」「训练方案」「防坑清单」「验收标准」执行。

开工前必须守住的纪律（违反会浪费大量算力或产出不可信数字）：
1. ~~先加载 skill：kaggle-vision-pipeline-guards~~ → **该 skill 不存在**，防线已内联实现在 `src/vision/guards.py`，直接用 `python -m src.vision.guards --self-test`。
2. 必须先 `python scripts/p1_check_env.py` 全绿、再 `python scripts/p1_fetch_dataset.py --dir D:/datasets/fdm-defect` 校验 ok:true，才能开训。
   - ✅ **不需要配 Kaggle 凭据**（DC-02R 是公开数据集，匿名可下）——见第 4 节实测结论。
3. Off_platform 只有 91 张 → 必须报 5 折 macro-F1 均值 ± 标准差，绝不可只报某一折。
4. 类别映射：Warping/Stringing/Layer_shifting 同名直接映射；Cracking、Off_platform 须逐条核对知识库后再定，写报告标注核对依据。
5. 不要改 P0 检索/生成层（已严格验证到顶，重做只会回归）；P1 只动 src/vision/。
6. **禁止 `timm.create_model(..., pretrained=True)`**（Windows 符号链接 + hf_hub 截断 → 0 字节权重），一律用 `src/vision/weights.py` 的 `build_model()`。
6. 数字做完才填；对外口径标注「qwen-turbo 判官，不确定度约 ±7pp」；不把论文 87% 说成自己的结果。
7. 原始图像与权重（*.pth/*.pt）不进 git；只提交代码、报告、折划分 json、data/p1_dataset_manifest.json。

第一步：先确认环境与数据是否就绪，遇到阻塞先报告再行动，不要盲目开训。
```

### 接法 B：自动加载项目记忆（辅助，省去每次粘贴）

本仓库已提交 `.workbuddy/memory/MEMORY.md`（P1 浓缩上下文）。新机器**把克隆下来的仓库根目录直接作为 WorkBuddy 工作区打开**时，该文件会被自动注入，Agent 无需提示词也知道 P1 任务与红线。

⚠️ 前提：必须「以仓库根目录为工作区」打开；若把仓库放进别的目录当子文件夹，自动注入路径会错位，此时退回接法 A。

### Agent 接手后别踩的死路（已证伪，别复验）

| 死路 | 结论 | 来源 |
|---|---|---|
| 改 P0 检索（top-k=10 / rerank / query 改写 / 索引变体） | 全部证伪，净增益≤0 | exp/exp_retrieval_headroom.py + 路线表 |
| 生成层「重写式」改造（v2.3 / v2.8 / v2.8b） | 不过零回归安全门 / 等价 | exp/generation_layer_campaign_conclusion.md |
| 单折报指标 | Off_platform 仅 91 张，单折抖动 ±10pp | HANDOFF 第 4/5 节 |
| 拿 Kaggle README 的 Cracking=427 对下载 | 笔误，实为 472 | HANDOFF 第 2 节 |
| 把论文 Swin-T 87% 当自己对标 | 任务/数据集不同，只可参照不可比 | HANDOFF 第 3 节 |

### P1 完成判据（WorkBuddy 自测用）

5 折 macro-F1 ≥ 0.80（报均值±标准差）｜ 最弱类 recall ≥ 0.70 ｜ 5 类→KB 映射 100% 覆盖含依据 ｜ ≥5 失败案例带图 ｜ seed+折划分落盘可复现。全部达成即 P1 完成，push 回仓库。

---

## 10. 🔴 D8/D9：外部验证与正常类（2026-10-06，最新，务必先读）

> 这一节是 MEMORY.md 里承诺的「踩坑全集」所在地。**只测了 DC-02R 就上线是本项目最大的风险来源。**

### 10.1 外部数据集选型（已定）

| 数据集 | 结论 |
|---|---|
| **DC-04** Zenodo 18628777，CC BY 4.0（**需署名**） | ✅ **采用**。11.76 GB / 40,282 张 |
| 3D-ADAM | ⛔ 任务不匹配 |
| Garuda/JITK 2026（99.68%） | ⛔ 高到不可信 |
| DC-07 Kaggle `padraigvalenti`（CC0，83 MB） | ⛔ **不可用**：网络爬取含图库/渲染图，且仅 fail/success 二分类 |

**DC-04 实测只3 类**（台账原记 4 类有误）：`layershift_error_print` 6,700/
`extrusion_error_print` 7,006 / `no_error_print` 6,435，**无 stringing**。
两视图 `cropped/`(1080×1080) 与 `full_res/`(1920×1080)，**同名文件 = 同一场景**（2,150 个场景）。
场景 = **Prusa i3 + 红色立方件 + 工业相机**，与 DC-02R 域差异极大。

### 10.2 D8 结论：负结果（勿重跑）

`exp/d8_external_valid.py` → `exp/d8_external_result.md`

| 指标 | 值 |
|---|---|
| cropped 分类准确率 | **6.25%** |
| full_res 分类准确率 | **8.25%** |
| 5 类随机基线 | 20% |
| 预测最大类（`Stringing`）占比 | **86.6%** |
| **真值正常件**被判 Stringing | **91.0%** |
| 正常件误报率 | 72%~75% |

**形态是「输出塌缩」而非「识别变弱」**——输出与图像内容脱钩，置信度还很高（mean≈0.90）。

✅ **已排除工程 bug**：DC-04 图像像素正常；**同一条 `diagnose()` 管线在 DC-02R 上 200/200 全对**；
zip CRC 全过。→ **真实的域外泛化失败**。

**根因**：冻结骨干 + 线性探针**假设测试分布 = 训练分布**。

⚠️ **裁剪带实验在 DC-04 上因塌缩失效**（cropped 反而低 3.5pp），
**DC-02R 内部的裁剪带结论未被外部复现**——报告时必须带上这个限定。

### 10.3 D9 正常类：代码就绪，但**默认关闭**

**代码全部实现**（`src/vision/infer.py` + `pipeline.py`）：
- `CLASSES = DEFECT_CLASSES(5) + [No_error]`，只有分类头需重建
- `pipeline.py` 有 `No_error` 提前分支：不查 mapping、`kb_ids=[]`、
  `rejected=False`、`is_normal=True`
- 🔴 `NORMAL_CLASS` **追加在末尾**（改前 5 类顺序会让旧部署件与映射层索引失效）
- 🔴 **正常类不做 KB 映射**——知识库里没有「未发现缺陷」条目，硬造即为伪造映射

🔴🔴 **但 `NORMAL_TRAIN = 0`，默认不混DC-04 正常件。实测原因：**

| 测量项 | 实测值 |
|---|---|
| DC-04 三类质心 vs DC-02R 全体质心余弦 | **0.53/ 0.55 / 0.53** → 三类几乎重合 |
| 正常件 vs 缺陷件 余弦 | 0.698 / 0.657 |
| **缺陷类之间**余弦 | **0.663** → **比「正常 vs 缺陷」还接近** |
| 正常件相对缺陷类的领先 | **仅 +0.019** |
| `class_weight="balanced"` 放大 | **×1.76** |

⇒ 模型学到的是「**像 DC-04 ⇒ 正常**」，不是「**无缺陷 ⇒ 正常**」。
**DC-04 上 100% 的图都被报「未发现缺陷」**（含两层缺陷件，p≈0.999）。

🔴 `class_weight=None` **毫无改善**（同样 100%）→ **不是权重问题，是特征空间不可分**。
🔴 **调阈值也救不了**——塌缩下高置信度是危险信号，不是优势。

⚠️ **这比 D8 原塌缩更危险**：原「都判 Stringing」至少还会给排查建议；
全判正常则**一条建议都不给**，用户以为件是好的。

### 10.4 🔴 本项目最贵的一条教训

**D9 的 A/B 实验测了「正例」，没测「反例」。**
它验证了「DC-04 正常件能被认出 100%」，
**从未验证「DC-04 缺陷件不会被一起认成正常」**——而后者才是上线后最致命的。
**少测的那一项，让整个方案在线上反噬。**

🔴 **纪律：任何「新增一类」或「加一个判别分支」的方案，
必须同时测两个方向——新类不被旧类吞、旧类不被新类吞。只测一个方向等于没测。**

🔴 讽刺的是 `exp/d9_normal_class_design.md` **第七节 P1 探针已经设计了这个测试**
（「若缺陷件 5 类召回 <40% → 只能减少误报」），**但落地时没有执行**。
→ **设计文档写了探针 ≠ 探针被跑了；A/B 实验的「通过」不能替代它。**

### 10.5 新增护栏（已反向自检，真会 FAIL）

`src/vision/infer.py::_sanity_check_ood()`：
落盘前用 DC-04 缺陷件做健全性检查，
**误判成正常 ≥50% 直接抛 `InferError` 拒绝写文件**（判据先写死再跑，避免看到结果再挑说法）。

实测验证：
- `--n-normal 200` →✅ 被拦，**被拦目录零文件写入**
- 手动构造 100% 误判的分类器 → ✅ 抛 `InferError`

### 10.6 🔴 第 8 个坑：护栏自己也会说谎

分类头自检原本拿模块级 `CLASSES`(6) 当基准。
正常类默认关闭后分类头本就该是 5 类，自检却报「训练数据缺类别」——
**护栏在正确配置下误报**。
修法：基准改为 `classes_eff = DEFECT_CLASSES + ([NORMAL_CLASS] if n_normal_used else [])`；
`load_deployed_model` 把**实际类别**透出给 pipeline（5 类部署件 proba 只有 5 列，
按 6 类索引会 IndexError 或静默错位）。

🔴 **教训：护栏的判据也必须覆盖所有合法配置。**
一个只会 FAIL 的护栏，早晚会在正确用法上误报，然后被当成噪音关掉。

### 10.7 口径核对：A/B 与部署不一致（值不值做？值）

`exp/d9_class_weight_check.py`，实测（N=200 / 6 类 / group-aware折外）：

| 口径 | DC-02R 5 类 macro-F1 | 正常件 | 缺陷件保留 |
|---|---|---|---|
| `none`（A/B 用的） | 99.60% | 100% | 98.7% |
| **`balanced`（部署件）** | **99.73%** | 100% | 98.5% |

→ 保留 `balanced`（保住 91 张的 `Off_platform` 小类）。
🔴 顺带修正 A/B 的副产物：它报的「加类 −0.08pp」是**无 balanced 口径**测的。

🔴 **折外结论不能直接搬到全量部署件**：
折外时正常件被 held-out，部署件是**全量重训**、正常件被反复见到 → 行为显著不同。
这是「部署件高置信度是自证」的另一面。

### 10.8 回归验收现状（全绿）

| 项 | 结果 |
|---|---|
| 部署件 | 1,912 张 / **5 类** / `normal_class_enabled=False` |
| DC-02R 200 张 | 分类 **200/200**、判正常 0、拒答 0、KB 全有（自证 sanity，**非性能指标**） |
| `--n-normal 200` | ✅ 被护栏拦住，**目录零文件** |
| `python -m src.vision.guards --self-test` | ✅ 全绿 |
| `exp/p1_p0_wiring_check.py` | ✅ **ALL PASS**（A~F） |
| 拒答阈值 | 0.90（`infer.py` CLI 原默认 0.60，**已修正为 0.90**） |

### 10.9 后续优先级（别做无效功）

1. ⛔ **换大骨干 / 3B QLoRA 不解决域问题**（已实测否掉，算力不是瓶颈）
2. 🔴 **自采目标域数据**（正常件 **+** 缺陷件都要），换机型/材料必须重采并重标定
3. 🔴 **引入 OOD / 域检测，且必须先于分类器**——
   域外输入应先判「是否在本域内」，而不是硬给一个高置信缺陷类
4. 🔴 知识库缺口：`Blobs`/`Zits`/`Z Seam`/`Overhang` **三标签完全无图像** → 只能自采

---

## 11. P0↔P1 接线：现状与「为什么还出不了答案」（2026-10-06 复核）

### 11.1 链路已通，实测走通

```
图片 → P1（EffNet-B0 冻结 + 线性探针，5 类）
     → kb_v2 条目（primary + secondary + confusable）
     → P0 证据（KB 正文 + 根因）
     → P0 生成答案
```

实测（Cracking 图）：
- `kb_ids = ['KB009','KB049']` —— 与 `kb_mapping` **完全一致**
- `evidence[0] = KB009 "Layer Separation and Splitting"` + 根因正文 ✅
- `route=vision`、`rejected=False`、`layer=none` ✅
- 验收 `exp/p1_p0_wiring_check.py`：**A~F ALL PASS**

| 验收项 | 含义 |
|---|---|
| A | 无图 → 纯文本路径，P0 逐字段零回归 |
| B | 5 类映射证据都在 10,000 字预算内，KB 全保留、不触发二次截断 |
| C | 每张图的 kb_ids 与 kb_mapping 全量映射一致 |
| D | 视觉拒答时 evidence 仍非空（top-2 留给人工） |
| E | 两条路径的返回字段均齐备（契约完整） |
| F | LangGraph 每张图都真的走 vision 路由（抓「静默丢弃入参」那个bug） |

### 11.2 🔴 但「出答案」那步需要 API，本机没配

`VisionDiagnoseAgent.answer()` 实测崩在最后一步：

```
File "src/answer_generator.py", line 239, in _call_model
    import dashscope
ModuleNotFoundError: No module named 'dashscope'
```

且 `DASHSCOPE_API_KEY` **未设置**。

⚠️ **这不是接线 bug，是环境依赖。**
⚠️ 中文 query **必须**配 `DASHSCOPE_API_KEY`——
离线兜底 TF-IDF 对中文**双路皆零**（英文语料 + 默认 `token_pattern` 不分中文 → nnz=0）；
BM25 被同一分词器拖累 → `retrieve("打印件表面有拉丝")` 返回 KB001~005 **字母序**。

**要跑通需**：
```bash
.venv/Scripts/pip install dashscope
# 并设置环境变量 DASHSCOPE_API_KEY
```
验收脚本用 `use_judge=False` 绕过判官/生成，所以**验收能过但 demo 不能跑**——
这是两件事，别混。

### 11.3 层间分工（复核确认，非缺陷）

| 层 | 入口 | 返回 |
|---|---|---|
| Skill | `VisionDiagnoseSkill.run()` | **证据**（evidence / kb_ids / layer），**无 answer** |
| Agent | `VisionDiagnoseAgent.answer()` |证据 + **answer** |

🔴 查证据字段用 `evidence[].kb_id` / `.problem` / `.text`，
**不是** `.kb_ids` / `.title`（不存在，误查会拿到 None）。

### 11.4 顺手修的一处措辞

`skill_vision_diagnose.py` 拒答文案原写死「概率仅 {proba}」，
在验收 D 项（**故意**把阈值设成 1.01 触发拒答）会输出「概率仅 1.0000」——
概率 1.0 本该通过任何 ≤1.0 的阈值，这句话会让人误判阈值标定有问题。
已改为「概率 {proba:.4f} < 阈值 {th:.4f}」。
