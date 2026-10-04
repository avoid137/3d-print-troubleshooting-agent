# P1 方案与风险评估（本机实测复核版）

> 生成时间：2026-10-04 ｜ 机器：本机（RTX 4060 8GB，驱动 CUDA UMD 13.3）
> 依据：实测 `nvidia-smi` / `pip` / 依赖探测 / `kb_v2.json` 逐条查询，**不是照抄 HANDOFF**
> 结论一句话：**技术方案成立且算力充裕，但有 1 个硬阻塞（Kaggle 凭据缺失）和 3 处必须修正的环境偏差；此外 HANDOFF 漏了 1 个会污染指标的真实风险（近重复图像泄漏）。**

---

## 一、结论速览

| 判断项 | 结论 |
|---|---|
| 方案**可行性** | ✅ 成立。1912 张 / 5 类小样本 + ImageNet 预训练是标准迁移学习场景 |
| 算力**够不够** | ✅ 大幅过剩。5 折 × EfficientNet-B0 全程 ≈ **20~30 分钟**（详见 §四成本） |
| 有无**硬阻塞** | 🔴 有1 个：**Kaggle 凭据完全不存在**，数据集无法下载 → 训练无法启动 |
| HANDOFF 是否可直接执行 | ⚠️ 不可直接执行，有 3 处偏差 + 2 处遗漏（§二、§三） |
| 最大技术风险 | 🔴 **近重复图像导致 5 折交叉验证泄漏，指标虚高**（§三.1，HANDOFF 未提） |

---

## 二、实测 vs HANDOFF：3 处必须修正的偏差

| # | HANDOFF 写的 | 本机实测 | 处置 |
|---|---|---|---|
| 1 | `pip install torch --index-url .../cu124` | 当前 Python **3.13.14**；cu124 索引下 cp313 只有 `torch 2.6.0 + torchvision 0.21.0`（老版本）；cu126 下有 `torch 2.9.1 + tv 0.29.1` | **改用 cu126**（§四方案 D-2）。cu124 在 py3.13 上会把torch 锁死在 2.6，timm 新版本可能不兼容 |
| 2 |「先加载 skill `kaggle-vision-pipeline-guards`」 | **该 skill 在本机不存在**（`~/.workbuddy/skills/` 为空，全盘搜索无结果） | 我按其描述的**五道防线内联实现**到 `src/vision/guards.py`，并在开训前自检防线本身会 fail（第五道防线要求） |
| 3 | 「C 盘仅剩 13 GB」→ 数据落 D 盘 | C 盘剩 **36 GB**，D 盘剩 **298 GB**，E 盘 466 GB | 仍落D 盘（`D:/datasets/fdm-defect`），纪律不变，但空间不再是风险 |

**另需注意（非偏差，是事实补充）**：

- **所有依赖当前一个都没装**：torch / torchvision / timm / sklearn / numpy / PIL / pandas / matplotlib / kaggle 全部 MISSING。也就是说 `scripts/p1_check_env.py` 现在跑必然 4 项 FAIL —— 这是正常起点，不是环境坏了。
- **显存已被占979 MiB**（Windows 桌面 WDDM，`explorer.exe` + `StartMenuExperienceHost`）。训练时 batch size 需留余量，峰值控制在 < 6 GB。
- **驱动 CUDA UMD 13.3 支持向下兼容**，cu126 运行时无问题。

---

## 三、风险评估

### 3.1 🔴 高风险：近重复图像导致交叉验证泄漏（本轮新发现，HANDOFF 未提）

**问题**：DC-02R 是同一批打印件连续拍摄的缺陷照片，同一实物可能有十几张近乎重复的图。若同一实物被分到训练折和测试折，模型等于**见过测试图**，5 折 macro-F1 会显著虚高，而且**虚高幅度未知**——这正是 P0 已经踩过的坑（判官间 ±7pp 噪声让 6.25pp 的增益无法判定）。P1 不能再引入一个说不清的量。

**处置**：开训前做感知哈希（pHash / dHash）近重复检测，对距离 < 阈值的图片**按组折叠到同一折**（group-aware split），而非逐张随机分层。成本：1912 张算哈希 + 两两比较，CPU 上不到 1 分钟。

**验收门**：若折叠后折间分布变化明显（同一实物跨折数 > 0），说明随机分层确实漏了，必须用 group-aware 划分；报告中**同时给出「普通分层」和「group-aware」两套 macro-F1**，差值就是泄漏幅度的量化估计——这本身就是可以写进报告的严谨证据。

### 3.2 🔴 硬阻塞：Kaggle 凭据缺失

`~/.kaggle/` 不存在，环境变量 `KAGGLE_USERNAME` / `KAGGLE_ACCESS_TOKEN` 均无。`scripts/p1_fetch_dataset.py` 会直接 `sys.exit(3)`。

**需要你提供**（三选一，密钥不要贴进对话、不要写进仓库）：
- ① `~/.kaggle/kaggle.json` = `{"username":"<你的>","key":"<API key>"}`
- ② 环境变量 `KAGGLE_USERNAME` + `KAGGLE_ACCESS_TOKEN`
- ③ 从旧机器拷 `~/.kaggle/access_token` 到本机同路径

**在凭据到位前，仍可推进的工作**（不依赖数据）：环境安装、代码骨架、合成数据冒烟测试、防���自检。这些做完，数据一到就能直接开训。

### 3.3 🟡 中风险：Cracking 的映射存在**概念混淆**风险

HANDOFF 说 Cracking「需逐条核对知识库」，但没给答案。我查了 `kb_v2.json`（56 条），相关条目有两个，**它们是不同的失效模式**：

| KB 条目 | 现象描述 | 根因数 | `img_classes` 标注 |
|---|---|---|---|
| `KB049` | Cracks Have Appeared in Tall Objects | 2 | `Delamination` |
| `KB009` | Layer Separation and Splitting | 8 | `Delamination` |

**风险**：若把 Cracking 一对一硬映射到其中一条，会把「表面/内部裂纹」和「层间分离」混为一谈；而这两个在 DC-02R 里是**不同的类**，混淆会直接让下游诊断给出错误根因。

**我的判断**：`Cracking` 主映射应给 `KB049`（唯一以「裂纹」为现象主语的条目），`KB009` 作为**次要/易混淆候选**保留并在映射表里显式标注冲突。**但最终必须看图确认**——DC-02R 的 Cracking 具体拍的是哪种裂纹，文档里查不到，只能下载后抽样人工看。这条不能凭想象定。

### 3.4 🟡 中风险：Off_platform 的映射同样待看图确认

`kb_v2` 里候选有4 条：`KB002` Not Sticking to the Bed(18 因)、`KB028` First Layer Issues(13 因)、`KB035` Nozzle Too Close to Print Bed、`KB037` Print Head Misses the Bed。

**特别注意**：`KB101/102/103` 是 Spaghetti（喷不出料成一团丝），与 Off_platform（打印件脱离平台）**不是同一现象**，虽然旧 `mapping.md` 把 Spaghetti 和 Stringing 的成因混淆过（该文件第 78 行自己承认了这个坑）——不要重复这个错误。

同样需下载后看图确认 Off_platform 拍的是「整体离床」还是「首层未粘」。

### 3.5 🟡 中风险：`data/mapping.md` 已失效，不能作为映射层依据

核查发现：该文件基于已弃用的 8 类体系，且它引用的 **45 个 KB ID 中有 8 个在 `kb_v2.json` 中不存在**（`KB029/KB030/KB045/KB059/KB066/KB068/KB070/KB073`——这些只在 `kb_v1` 的 75 条里）。若照抄它做映射，会引用不存在的条目。

**处置**：映射层**基于 `kb_v2.json`（56 条，唯一在用版本）从零重建**，并把旧文件顶部的弃用警告保留、加强。

**好消息**：`kb_v2` 已有 20/56 条填了 `img_classes` 字段（`Stringing`/`Layer Shifting`/`Warping`/`Delamination` 等旧 8 类词汇），可作为 3 类同名映射的**现成锚点**，不必全靠人工判断。

### 3.6 🟢 低风险：`scripts/p1_check_env.py` 有一处会误判

脚本第 92行 `match = (counts == exp)` 要求类别计数**完全相等**才通过。若 Kaggle 包内多出 `.png`/`.jpeg` 或有隐藏目录，会误报 FAIL 而非真实问题。**处置**：放宽为「每类 ≥ 期望值且总量 ≥ 1912」即 PASS，精确计数另行打印——避免因为一个无关文件卡住整条流水线。

---

## 四、方案（D0~D4 五阶段，每阶段有验收门）

###阶段总览

```
D0 凭据与依赖├─ 环境安装（cu126）→ p1_check_env 除数据外全绿
D1 管线防线     ├─ guards.py 五道防线 + 合成数据冒烟（不需要数据）
D2 数据         ├─ 下载 6.3GB → manifest ok:true → pHash 近重复分组
D3 训练         ├─ 5 折 × EfficientNet-B0 → macro-F1 均值±标准差
D4 映射与报告   ├─ 看图核对 2 类 → 映射表重做 → exp/p1_vision_result.md
```

**关键设计：D0/D1 不依赖数据，D2 需要凭据。** 所以在你配凭据之前，我可以把 D0/D1 全做完。

### D0 · 环境（关键修正：cu126）

```bash
cd F:/workbuddy/3d-print-troubleshooting-agent
python -m venv .venv && .venv/Scripts/activate
# ⚠️ 用 cu126，不是 HANDOFF 的 cu124 —— py3.13 下cu124 只有 torch 2.6/tv 0.21
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-p1-vision.txt
python scripts/p1_check_env.py     # 预期：除「数据集」外全绿
```

**验收门**：`torch.cuda.is_available() == True` 且打印 `NVIDIA GeForce RTX 4060`。

### D1 · 管线防线 + 冒烟（不需要数据集）

新增 `src/vision/guards.py`，把 HANDOFF §4 的五道防线落成代码：

| 防线 | 实现要点 |
|---|---|
| 1 骨干审计 | 打印实际骨干名 / 参数量 / `timm.resolve_data_config()` 的 mean·std，**禁止手写归一化常数** |
| 2 目录层级探测 | 抽样报告每类命中率，< 0.5 硬失败 |
| 3 运行时计数器 | 累计空样本 / 读取失败；空样本 > 5% 立即中止（先查管线，别改模型） |
| 4 无数据冒烟 | `--smoke-test` 用合成数据跑通全链路（64px、1 epoch、CPU 可跑），抓形状 bug |
| 5 防线自检 | 每条断言都要先验证「条件不满足时确实会 fail」——永远通过的检查是装饰 |

**为什么要认真做**：这类代码最典型的失败模式是「数据根本没喂进去，但 loss照降、分数就是不高」，不报错的错误最贵。

**验收门**：`--smoke-test` 全链路跑通；**并额外做一次反向验证**：故意把数据目录指错，确认防线 2 真的 FAIL（否则防线是装饰）。

### D2 · 数据（依赖凭据）

```bash
python scripts/p1_fetch_dataset.py --dir D:/datasets/fdm-defect
```
**验收门**：`data/p1_dataset_manifest.json` 的 `ok: true`（修正 §3.6 的误判后）。

**追加一步（新增）**：pHash 近重复检测 → 生成 `data/p1_folds.json`，含两种划分：
- `folds_stratified`：普通分层 5 折（seed=42）
- `folds_grouped`：按近重复组折叠的 5 折（seed=42）

### D3 · 训练

| 决策 | 选择 | 理由 |
|---|---|---|
| 骨干 | **EfficientNet-B0**（timm，ImageNet 预训练） | 小数据集上大模型必过拟合；项目定位轻量可部署 |
| 对照 | MobileNetV3-small | 验证「更小的骨干是否够用」，也是部署成本的参考 |
| 输入 | 224×224 | 缺陷是纹理/几何形态，224 足够；不够再试 288 |
| 划分 | 分层 5 折 seed=42 **×两种划分（普通/去泄漏）** | §3.1 |
| 不平衡 | 类别加权 + 轻量采样 | 538:91 ≈ 6:1 |
| 主指标 | **macro-F1 均值 ± 标准差** + 混淆矩阵 + 各类recall | Off_platform仅 91 张，单折测试集约 18 张，**单折抖动可达 ±10pp** |
| 增强 | 水平翻转 + 轻色抖 + 随机裁剪；**不做强几何扭曲** | 打印件有方向性，强扭曲造伪样本 |
| 训练量 | 每折 ≤ 30 epoch，早停 patience 5 | |

**明确否掉的方案**：`00_项目准备总览.md` 提到「本地微调 Swin-T / Qwen2.5-VL-3B (QLoRA)」。
**否掉理由**：① 验收指标是分类准确率，轻量CNN 完全够；② 8GB 显存做 3B QLoRA batch 只能 1~2，训练极慢且极不稳；③ 依赖重（transformers + qwen-vl-utils）、模型下载 6GB+，性价比极低。**若后续要做零样本对照，用 qwen-vl API 调用即可，不必本地微调。**

**验收门**：macro-F1 ≥ 0.80（两组划分都报，差值即泄漏幅度）、最弱类 recall ≥ 0.70、fold 划分落盘。

### D4 · 映射层+ 报告

1. **看图核对**（不可跳过）：从Cracking 抽 10~15 张、Off_platform 抽全部 91 张人工看图，确认现象归属→ 定稿映射。
2. **基于 `kb_v2.json` 重做映射表**（5 类 100% 覆盖），每条标注 `match` 类型与**人工核对依据**（看了几张图、结论是什么）。
3. **≥ 5 例失败案例**（带图 + 误判去向），这是 PMF 材料。
4. 报告 `exp/p1_vision_result.md`，写明种子、划分方式、泄漏幅度。

---

## 五、成本估算

| 阶段 | 耗时 | 说明 |
|---|---|---|
| D0 环境安装 | ~15 min | torch cu126 轮子约 2.5 GB 下载 |
| D1 防线 + 冒烟 | ~30 min | 写代码 + 正反向自检 |
| D2 数据下载 | 10~40 min | 6.3 GB，视网速 |
| D3 训练（5 折×2 套划分） | **20~30 min** | 单折约 2 min，5 折 ×2 套 = 约 25 min |
| D4 映射 + 报告 | ~60 min | 含看图核对 |

**算力结论：4060 远够用**，瓶颈是数据到位与管线正确性，不是算力。这也说明「算力不够所以要上Swin-T/3B」是伪问题。

---

## 六、需要你拍板的 3 件事

| # | 决策点 | 我的建议 |
|---|---|---|
| 1 | **Kaggle 凭据**（唯一硬阻塞） | 你配`~/.kaggle/kaggle.json` 或环境变量，**不要贴进对话**。配好后告诉我，我继续 D2 |
| 2 | 是否先做 **D0+D1**（不依赖数据） | **建议先做**。装环境 + 写防线 + 冒烟测试，配凭据的时间可以并行省下 |
| 3 | 5 折是否跑**两套划分**（普通 + 去泄漏） | **建议跑**。成本仅多 10 分钟，换来「泄漏幅度」这个可写进报告的量化证据，也避免指标虚高 |

---

## 七、纪律红线（沿用 P0，不破）

1. 原始图像、权重（`*.pth`/`*.pt`）**不进 git**。
2. 只提交代码、报告、fold 划分 json、`data/p1_dataset_manifest.json`。
3. **数字做完才填**；未达标写「持续迭代中」。
4. 不把论文 Swin-T 87% 说成自己的结果（任务与数据集不同，不可直接比）。
5. P0 检索/生成层**一行都不改**（已验证到顶，重做必回归）。P1 只动 `src/vision/`。
6. 映射表**不硬凑**：Cracking / Off_platform 必须看图核对，报告标注核对依据。