# 3D 打印故障诊断 Agent · 项目记忆（仓库自动加载）

> 本文件随仓库 clone 到新机器；把仓库根目录作为 WorkBuddy 工作区打开时自动注入。
> 完整接手文档始终是仓库根的 **HANDOFF_P1.md**（权威、面向 Agent 与人都可读）。

## 当前主线：P1（视觉分类，RTX 4060 机器）

- **P0 已交付且严格验证到顶**（别再碰）：strict SOTA **54.37%**(87/160, qwen-turbo 判官，判官间一致率 74%→不确定度约 **±7pp**)、lenient 69.37%、拒答误拒 0。检索侧（top-k=10 / rerank / query 改写 / 索引变体 8 个）与生成层（v2.3 净负 / v2.8 +2.5pp 但 23 回归 / v2.8b 等价）全部证伪或不过零回归安全门。结论见 `exp/generation_layer_campaign_conclusion.md`。
- **P1 任务**：5 类缺陷图像分类（Kaggle `wengmhu` / DC-02R，MIT 许可，1,912 张）+ 视觉→知识库映射层。
- **P1 只动 `src/vision/`**，不要改 P0 检索/生成层（重做必回归）。

## ✅ P1 核心结论（2026-10-04 实测定稿，数字勿改）

**选定方案 = L2 冻结骨干 + 线性探针，零训练成本。**

| 级别 | EfficientNet-B0 | MobileNetV3-small |
|---|---|---|
| L1 冻结+kNN | 98.44 ± 0.57% | 98.16 ± 0.80% |
| **L2 冻结+线性探针** | **99.73 ± 0.06%** | **99.41 ± 0.70%** |
| L3 全量微调 | 99.19 ± 0.98% | 99.25 ± 0.48% |

（group-aware 折，5 折 × 2 种子 = 10 次/格，src：`exp/p1_d3_summary.json`）

- **L3 全量微调无增益**（−0.54pp / −0.16pp，两骨干复现），且 B0 上σ 是 L2 的 16 倍。
  L2 提特征 40s + 一次 LogReg **秒级**；L3 要20 次微调（12~13 min）换负收益。**别再跑 L3。**
- **两骨干完全等价**（差 0.06~0.31pp）→ 本数据集已触及「ImageNet 预训练 + 轻量分类器」上限。
  **「算力不够所以上 Swin-T / 3B QLoRA」这个伪问题已有实测依据。**
- **泄漏幅度六格全部 <1pp** → 可忽略（折叠仅 163/1912=8.5%；普通分层实际跨折 234 张=12.24%）。
- 🔴 **对外口径**：99.73% 只能说「DC-02R 内、group-aware 折、5 类 macro-F1」，
  **不能**说「缺陷识别准确率 99.7%」或「SOTA」。理由见 `exp/p1_vision_result.md` §2.4。

## 🔴 本项目最重要的一条方法论纪律

**判定「模型是否走了捷径」不能只看「该线索单独能到多少分」，
必须主动移除该线索、看分数掉多少。**

实例：DC-02R 仅用 19 维低级像素统计（无空间结构）就能拿 71.5% macro-F1，
各类背景色两两分明（Warping/Off_platform 绿床 vs 其余灰白）→我据此**推断**
「99% 反映的是识别拍摄批次而非缺陷形态」。**该推断被干预实验推翻**：
去背景 0.00pp、去色+去背景仅 −0.33pp。低阶捷径客观存在但线性探针并不依赖它。
两者结论可以完全相反。脚本：`exp/audit_d3_credibility.py` + `exp/d3_intervention.py`。

## 新机器接手顺序（必做）

> ⚠️ 2026-10-04 在 4060 机器实测复核，以下第 1、2 条已修正，详见 `exp/p1_plan_review.md`。

1. ~~加载 skill `kaggle-vision-pipeline-guards`~~ → **该 skill 在本机不存在**（全盘搜索无结果）。改为把五道防线**内联实现**到 `src/vision/guards.py`，并按第五道防线要求做「反向自检」：故意指错数据目录，确认防线真的会FAIL。
2. 环境：建 `.venv` → ⚠️ **改用 `cu126`**（`--index-url https://download.pytorch.org/whl/cu126`）。本机 Python 3.13，cu124 索引下 cp313 只有 torch 2.6/tv 0.21（老版本）；cu126 有 torch 2.9.1/tv 0.29.1。4060 需 CUDA 版，装 CPU 版不报错但慢 20×。→ `python scripts/p1_check_env.py` 除「数据集」外全绿。
3. 数据：✅ **已完成**（2026-10-04）。数据根 = `D:/datasets/fdm-defect/FDM-3D-Printing-Defect-Dataset/data`（**两层嵌套**，别只 iterdir 一层）。张数 538/472/447/364/91=1,912 全部核对一致，30 张抽样 0 损坏，尺寸统一 3072×2048。`p1_check_env.py` 5/5 全绿。⚠️ Kaggle README 的 Cracking=427 是笔误，实为 472。
4. ✅ **训练已完成**（2026-10-04）。EfficientNet-B0 + MobileNetV3-small双骨干 × 三级对照 × 双口径 × 2 种子 = 60 次评估。**结论见上方「P1 核心结论」**。复现命令：
   - `python -m src.vision.dedup --data-root D:/datasets/fdm-defect`（D2，约 3 min，特征走缓存）
   - `python -m src.vision.train --level L2 --backbone efficientnet_b0 --scheme both --seeds 42,43`（约 4 min）
   - `python exp/summarize_d3.py`（统一汇总 + 一致性断言）
5. ✅ **pHash 去泄漏已完成并定稿**（2026-10-04）。产出 `data/p1_folds.json`：1,749 group、
   跨类污染 0、前缀 GT 召回 100%。🔴 **pHash/dHash 已彻底降级为只读诊断、不参与 union**
   （实测误合并率 94.3%，且传递闭包产出 19 个跨类污染 group）。**别回退。** 详见
   `src/vision/dedup.py` 模块 docstring + 报告 §1.2.0。

## 已知阻塞与坑（2026-10-04 实测）

- ✅ **原「Kaggle 凭据缺失」已作废 —— 不需要凭据**。DC-02R 是**公开**数据集，匿名即可下载：元数据端点 `api/v1/datasets/view/wengmhu/fdm-3d-printing-defect-dataset` 无认证返回 **200**，`kaggle datasets download` 无凭据直接开始传 5.85 GB。**别再让用户配凭据了。**
- ✅ **原「预训练权重下不来」已解决** —— 根因不是网络，是 **Windows 无符号链接权限 + hf_hub 传输截断**（详见下方「权重加载」节）。已固化为 `src/vision/weights.py`。
- ⚠️ **网络可达性实测（关加速器 + `unset *_PROXY` 后）**：
  - ✅ `hf-mirror.com` 200 @2.56s，稳定 1.0 MB/s —— **唯一可用的权重源**
  - ✅ `kaggle.com` 200；`storage.googleapis.com`（Kaggle CDN）**支持 HTTP Range** → 可分片并发
  - ❌ `huggingface.co` 超时（官方域名被墙）
  - ❌ **`modelscope.cn` 魔搭 status=000 不可达**（3 次重试全失败）—— **不必再试**；且 timm 权重在魔塔无自动对应镜像路径
  - ❌ `mirrors.aliyun.com` status=000
  - ⚠️ `zenodo.org` 需 `curl --resolve zenodo.org:443:<真实IP>` 绕 DNS 污染（直连 302 到 m.baidu.com），通但仅 16 KB/s → DC-04 外部验证暂缓
- 🔴 **权重加载铁律（`src/vision/weights.py`，别绕过）**：
  - `header too small` ≠ 网络失败。实测 blob = 21,355,344 B **完整**，snapshot = **0 B**（符号链接失败）。hf_hub 还会把传输截断成 0 字节（`local_dir=` 也一样），而 `curl` 同一 URL 完好。
  - **禁止 `timm.create_model(..., pretrained=True)`**（guards.py / train.py / dedup.py 三处已全部换成 `weights.build_model`）。torch.hub 的 DINOv2 同样不可用。
  - `load_pretrained_into` 会**校验注入张量数**，注入 0 个或不足一半即抛 `WeightError`——防静默退化成随机初始化。
  - `_read_state_dict` 用**魔数嗅探**而非后缀（HF blob 是 sha256 无扩展名，内容却是 safetensors；按 `.pth` 走 torch.load 会报 `UnpicklingError: Unsupported operand 176`）。
  - `EXPECTED_BYTES` **全为实测值**：B0 = 21,355,344；DINOv2 = **346,334,872**（最初按估计写 346,584,396，差 249 KB，会把完好文件误判为截断而反复重下）。
  - 权重落 `data/_weights/`，**已加 .gitignore**（`*.safetensors` + `data/_weights/`），368 MB 不入库。
- 💡 **Kaggle 并发下载实测**：1 线程 ~250 KB/s（6.5 h）→ **8 线程 ~697 KB/s（2.4 h）** → 16 线程无增益（带宽封顶）。签名 URL 有效期 3 天（`X-Goog-Expires=259200`），断了重取 URL 续传。
- 🔴 **Kaggle `totalBytes` 元数据不可作为下载完整性基准**（2026-10-04 实测）：元数据报 6,293,272,351 B，但 Google Storage 实际对象 `x-goog-stored-content-length` = **6,279,686,097 B**，差 13,586,254 B。**这是 Kaggle 侧元数据误差，不是下载缺失**——按元数据去补下会被服务器拒为 `InvalidRange`。
  - **正确判据**：zip 条目数 = 1,912 + `zipfile.testzip()` CRC 全过 + 5 类张数一致。**与元数据较劲没用，结构化校验才是可靠判据。**
- ⚠️ **并发下载别拿短时峰值当结论**：8 线程短时实测 697 KB/s，但全程 5 h 09 min 均速只有 **~440 KB/s = 单线程 1.75×**（非 2.8×）。长跑会掉速，且最后一片始终单线程收尾。
- ⚠️ **别用 `except: pass` 掩盖导入失败**（2026-10-04 实踩）：修「自检脚本数到 0 张」时用 `try: from src.vision.guards import ... except: pass`，但脚本在 `scripts/` 下 `sys.path[0]` 是 `scripts/` → 导入失败被吞 → 下探静默失效、依然报 0 张，白排查半小时。正确做法：`importlib.util.spec_from_file_location` 按路径加载 + 显式 `sys.modules` 注册（dataclass 解析需要）。
- 🔴 **七道防线（`src/vision/guards.py`，`--self-test` 全绿，累计抓到 3 个真 bug）**：
  1. 骨干审计（归一化只取 `resolve_data_config`）
  2. 目录层级探测（`probe_dataset` 自动下探，命中率<0.5 硬失败）
  3. 运行时计数器（空样本>5% 中止；`record_empty`/`record_failed` 语义必须分开）
  4. 无数据冒烟（`train.py --smoke-test`）
  5. 反向自检（`assert_can_fail`）
  1'. **权重加载**（`load_pretrained_into` 校验注入张量数，0 个即抛错）
  7. **AST 静态符号检查**（`check_undefined_symbols`，抓「删 import 忘补」这类语法合法的 NameError）
- ⚠️ **依赖初始全空**：torch/timm/sklearn/numpy/PIL/pandas/matplotlib/kaggle 一个都没装，`p1_check_env.py` 初始 4 项 FAIL 属正常起点。
- ⚠️ **`data/mapping.md` 已失效，不可作映射层依据**：基于弃用的 8 类体系，引用的 45 个 KB ID 中有 8 个（KB029/030/045/059/066/068/070/073）不在 `kb_v2.json` 里（只在 kb_v1 的 75 条）。**映射层基于 `kb_v2.json`（56 条）从零重做。**
- 💡 kb_v2 已有 **20/56 条填了 `img_classes`**，可作 3 类同名映射的现成锚点。
- ⚠️ **Cracking 别硬映射**：kb_v2 里 `KB049`（Cracks Have Appeared in Tall Objects）与 `KB009`（Layer Separation and Splitting）是不同失效模式、都标 `Delamination`。倾向主映射 KB049，KB009 标易混淆候选，**但必须下载后看图定稿**。
- ⚠️ **Off_platform 别和 Spaghetti 混**：KB101/102/103 是 Spaghetti（喷不出料成团丝），与「打印件脱离平台」不是同一现象。旧 mapping.md 曾把 Spaghetti 与 Stringing 成因混淆，别重复。
- 💡 **算力不是瓶颈**：5 折 × 2 套划分 = 10 次训练，4060 全程约 20~30 分钟。「算力不够所以上 Swin-T / 3B QLoRA」是伪问题，已明确否掉（8GB 跑 3B batch仅 1~2、极慢不稳）。零样本对照用 qwen-vl API 调用即可。

## P1 方法与数据集定案（2026-10-04 第二轮，实检证据）

- 🔴 **去泄漏必须双检，不能只用 pHash**：SSRN 7395570（5 seed×5 fold）用真实病灶 ID 验证，**pHash 仅召回 3.3% 真实同病灶组**。方案= pHash 粗筛(dHash≤5) → **DINOv2 冻结特征精筛**（补不同视角同实物）→ **阈值人工标注自标定**（不照抄 5）→ 连通分量聚组 → `StratifiedGroupKFold`。**group-aware 为主口径**，stratified 仅作对照算差值。泄漏影响是数据集特定的，必须实测。
- ✅ **训练做三级对照**（替代只跑全量微调）：**L1** 冻结骨干+kNN（同领域2026 论文同款，分钟级）／**L2** 冻结骨干+线性探针 DINOv2（分钟级，小样本常最优）／**L3** 全量微调 EfficientNet-B0（作上限）。每级跑两套折划分=6 组。组内差异 <2pp 判「等价」。
  - 同领域证据：Bast2026, IJAMT 144:2047-2059（DOI 10.1007/s00170-026-17844-w）ResNet50 特征+kNN，**加权 F1 仅 0.812**，单类 0.757~0.990。
  - 跨领域证据：植物病害 PlantDoc 上冻结 DINOv2 线性探针比全量微调 EfficientNet-B0 **+23.4pp**；全量微调使跨域 50.6%→29.4%。
  - ⚠️ 反例：字体 394 类细分类线性探针仅 36.6%（远低于 LoRA 98.7%）→ **线性探针必须实测，不能假设够用**。
- ✅ **DC-04（Zenodo 18628777）升格为 P1 外部验证集 + 正常件门控源**（2026-10-04 已补全台账：许可 **CC BY 4.0必须署名**、关联论文 DOI、GitLab 地址、该论文 baseline）。
  - 价值：解决 HANDOFF §6.5「第三方泛化验证需外部人力」→ **用另一公开数据集即可，无需外部人力**；含无缺陷件→ 可做正常/异常门控。
  - ❌ **不能替代 DC-02R 做主集**：Zenodo 4 类无 Warping/Cracking/Off_platform，**换了主集映射层就断**。只下 `test_data.zip`（12.6 GB）。
  - 💡 **无需 Kaggle 凭据**（Zenodo 开放）→ Kaggle 长期搞不定时的替代主集退路。
- ⛔ **排除**：3D-ADAM（NeurIPS 2025，RGB+3D 点云异常检测任务，任务不匹配）；Garuda/JITK 2026 的 99.68%（数字高到不可信，**不可作对标**）。
- 📏 **验收标准**：macro-F1 ≥0.80 保留但增设「同领域可达水平对照」（0.812 加权是同领域实测水位）；**必须报跨数据集差值**；建议 **5 折 × 2 种子**（5 折太少估不准标准差）。
- 💡 补两条：**多层特征融合**（浅层纹理+深层几何拼接，20 行代码）；**TTA 须在去泄漏划分上验证增益**，小于噪声报「等价」。
- 📄 详细方案：`exp/p1_plan_review.md`（风险评估）+ `exp/p1_risk_mitigation_and_alternatives.md`（改进+新数据集）

## P1 代码已就位（2026-10-04 第三轮）

| 文件 | 作用 |
|---|---|
| `src/vision/guards.py` | 五道防线（骨干审计/目录探测/RuntimeCounter/冒烟/自检）+ `--self-test` |
| `src/vision/dedup.py` | 去泄漏：dHash+pHash 粗筛 → DINOv2 特征精筛 → 并查集聚组 → `StratifiedGroupKFold`，产出 `data/p1_folds.json`（两套划分） |
| `src/vision/train.py` | 三级对照 L1(冻结+kNN)/L2(冻结+线性探针)/L3(全量微调)，每级 ×group-aware 划分 × 多种子；含多层特征融合；`--smoke-test` |
| `src/vision/kb_mapping.py` | 映射层权威源，**导入即校验 KB ID 存在**。已跑通：5/5 覆盖，产出 `exp/p1_kb_mapping.json` |

- ⚠️ **映射层 2 类待看图定稿**：Cracking →主`KB049`（易混淆 `KB009`层分离）；Off_platform → 主 `KB002`/`KB028`（易混淆 `KB101` Spaghetti、`KB067` 反向现象）。**禁止凭想象定稿。**
- 🔴 **本机有 HTTP 代理 `127.0.0.1:60318` 导致 Zenodo 下载不通**（证书中间人 + 节点无响应），但 pip 走同一代理可通PyPI → **DC-04 外部验证暂缓，换网络环境**。主线路 DC-02R 不受影响。
- 修`scripts/p1_check_env.py`：`counts == exp` 严格相等会误报 → 放宽为「每类≥基准且总量≥1912 即 PASS」。

## ⚡ P1 环境已就绪（2026-10-04 实测，别再重装）

```
torch 2.14.1+cu126 / torchvision 0.29.1+cu126 / timm 1.0.30（1351 模型）
CUDA True · RTX 4060 · 8.0 GB · GPU 冒烟峰值 0.29 GB
p1_check_env.py → 4/5 全绿（唯一失败项 = 数据集，等 Kaggle 凭据）
```
- ⚠️ **torch 实测是 2.14.1 不是 2.9.1**（cu126 索引已更新到更版本），requirements 文件已更正。
- ⚠️ 装包/下权重**必须先 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`**：
  加速器进程关掉后**环境变量仍残留**，会导致 `ProxyError 502`。
- 🔴 **预训练权重当前下不来**（本机对境外带宽极差）：HuggingFace 走代理 502 / 直连超时 /
  `HF_ENDPOINT=https://hf-mirror.com` 镜像下载不全（`SafetensorError: header too small`，清缓存重试仍失败）。
  **训练前必须解决**，否则只能随机初始化、效果差很多。
  审计逻辑已用 `pretrained=False` 验证通过：EfficientNet-B0 **4,013,953** 参数 / MobileNetV3-small **1,522,981** 参数。
- ✅ **五道防线全部自检通过**（`python -m src.vision.guards --self-test`），**并抓到 1 个真 bug**：
  `RuntimeCounter.record(ok:bool)` 把 ok=False 全记到 failed → `empty_rate` 恒为 0 → **防线 3 永不触发**。
  已改为 `record_empty()/record_failed()/record_ok()` 三个显式方法并分别检查两种失败率。
  **这正是「永远通过的检查即装饰」的实例** —— 若不做反向自检，缺陷会一直藏着。

## 纪律红线

- 原始图像 / 权重（`*.pth` `*.pt`）不进 git（`.gitignore` 已排除 `data/raw` `data/datasets`）。
- 数字做完才填；对外口径标「qwen-turbo 判官，不确定度约 ±7pp」；不把论文 Swin-T 87% 说成自己的。
- 跨材料口径一致；类别映射 3 类同名直接映射，Cracking / Off_platform 须逐条核对 KB 后定并标注依据。
- 只采用许可明确的源（DC-02R = MIT）；Unknown 不入库。数据集台账 = `data/dataset_cards.md`。

## 回迁

同仓开发直接 push；P1 产物放 `src/vision/`、报告 `exp/p1_vision_result.md`、折划分 json 与 `data/p1_dataset_manifest.json` 提交；回本机 `git pull` 同步。
