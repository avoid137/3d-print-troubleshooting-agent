# 3D 打印故障诊断 Agent · 项目记忆（仓库自动加载）

> 完整文档 = `HANDOFF_P1.md`；数字明细 = `exp/p1_vision_result.md`；逐日过程 = `.workbuddy/memory/YYYY-MM-DD.md`。
> 本文件只留「不能忘的」。**空间有限，细节一律回查上面三处。**

## 状态

P0 文本已到顶（strict SOTA 54.37%，**别再碰检索/生成层**）；P1 视觉99.73% 可信，
映射层 5/5 定稿，端到端+拒答+P0↔P1 接线全通。DC-04 外部验证数据已下完，**D8 是唯一未闭合项**。

## 纪律红线

数字做完才填 / P1 不改 P0 检索生成层 / 原始数据权重不进 git / 许可明确才用（DC-02R MIT、
DC-04 CC BY 4.0 需署名）/ **映射不硬凑，必须先看图** /
对外只能说「DC-02R 内、group-aware 折、5 类 macro-F1 99.73%」，**不能说「识别准确率 99.7%」或「SOTA」**。

## P1 数字（勿改）

**L2 冻结骨干+线性探针**：EffNet-B0 **99.73±0.06%** / MobileNetV3-s 99.41±0.70%。L1 98.4% /
L3 99.19%（**L3 无增益，别再跑**）。两骨干等价 → 已触「ImageNet 预训练 + 轻量分类器」上限，
**「算力不够上 Swin-T/3B QLoRA」是伪问题，已有实测依据**。泄漏六格全 <1pp。

**可信度三层递进**（每层判据必须**不可被前一层结论推出**）：混淆矩阵（错 6/1912，recall 0.9936~1.0）
→ 遮挡热区（跨类 r=**+0.015**）→ **裁剪带对照（下三分之一 99.91% 比全图还高，上带 91.44%）**
→ **模型确实在读缺陷件。** 局限：Stringing/Layer_shifting 遮挡r=+0.890 看的位置没区分开；
6 错例中 **2 例人眼更认同模型** → 真实上限 99.7~99.8%，**但报告仍报 99.69%，不擅自上调**。
⚠️ 只排除「认错东西」，**推不出「换机器/换材料也准」**。

## 映射层（看图定稿，改映射不需重训）

Stringing→KB006,KB063（conf KB101）｜Warping→KB024,KB044,KB016｜
Layer_shifting→KB008,KB050,KB052,KB065（conf KB009）｜**Cracking→KB009**（sec KB049）｜
**Off_platform→KB101,KB102,KB103**（sec KB002,KB028；conf KB067）

## 🔴 铁律与已修bug

**部署件高置信度是自证**：折外误判图（真值 Cracking/概率 0.6343）在部署件上给 **0.9928**（训练时见过）。
① 部署件**不能报 99.73%**（`infer.evaluate()`直接抛错拒绝执行）；
② **阈值必须用折外模型标定** → **拒答阈值 0.90**（正确 n=1906 mean 0.9957/min 0.5576；
错误 n=6 mean 0.6910 → 回答 98.6%、拦错 83.3%；拍脑袋的0.60 只拦 33% 且误伤）。⚠️ 高置信≠正确。

**P0既有 bug 已批准真修**（`load_fitted_embedder` 硬编码 cause 级路径 → kb 模式读不到 fitted，
用单条 query 重fit 迷你模型）。修法=按路径取+ `_FITTED_EMBEDDER_CACHE` 分桶 + `faiss_path` 透传+
`strict=True`；验收换成 `assert_no_bypass_needed()` **反向自检**。
🔴 **归因两次修正**：原写「抛 TruncatedSVD 越界」只对了一半——(a) 短 query → **抛错**；
(b) 长 query → vocab=3迷你模型查 16367 词表索引，**维度同为 1024 所以 FAISS 不报错**，
静默返假相似度（top1 KB002 cos=0.3613，正确 KB006）→ 危害更大。
⚠️ 修好后暴露独立缺陷（**未修**）：离线兜底 TF-IDF **对中文 query 双路皆零**（英文语料 +
默认 token_pattern 不分中文 → nnz=0 → 全零；BM25 被同一分词器拖累）→ `retrieve("打印件表面有拉丝")`
返回 KB001~005 **字母序**。中文**必须配 `DASHSCOPE_API_KEY`**（生产本就如此，不影响 54.37%）。

**P0↔P1 接线**唯一真问题 = `MAX_CONTEXT_CHARS=10000` 硬编码 vs 视觉证据 7.2k~28.3k，
贪心截断挤掉整块 → 改**max-min 轮转水填充**（5 类 KB 全保留、根因覆盖 54%，每处裁剪标注已截断条数）。
❌「按剩余体量比例分配」更差。验收 `exp/p1_p0_wiring_check.py` A~F 全过。
🔴 **抓到的 bug ①最危险：LangGraph 静默丢弃 schema 未声明入参**（`TypedDict(total=False)`）→
未声明的 `image_paths` 直接消失不报错 → 有图也走文本路径，返回值「看起来都正常」，
只能靠断言 `route=="vision"` 抓到。

## D8 外部验证（进行中）

`exp/d8_external_valid.py`，四组：①准确率（**只有 Layer_shifting 可算**）②正常件门控（应全拒）
③阈值敏感性 ④**裁剪带对照**（cropped 1080² vs full_res 1920×1080 同名=同场景两版本）。
🔴 显式标注 `UNCOVERED=[Warping, Cracking, Stringing]`，**绝不拿 1 类分数冒充 5 类泛化**。

## 环境与数据（已定）

torch 2.14.1+**cu126**（cu124 下 cp313 只有 torch 2.6；CPU 版慢 20×）/ tv 0.29.1 / timm 1.0.30，
RTX 4060 8GB，Python 3.13，venv=`.venv`。P0 索引原本不存在，需先 `python src/build_kb_index.py`。

- **DC-02R**（主集）Kaggle `wengmhu/...` MIT 1,912 张 3072×2048，
  根 `D:/datasets/fdm-defect/FDM-3D-Printing-Defect-Dataset/data`（**两层嵌套**）。
  **不需 Kaggle 凭据**（公开）。README 的 Cracking=427 是笔误，实为 472。
- **DC-04**（外部验证，**已下完**）Zenodo 18628777 CC BY 4.0，`D:/datasets/dc04/test_data.zip`
  11.76 GB / 40,282 张。🔴 **实测只3 类**（台账原记 4 类有误）：layershift 6,700 / extrusion 7,006 /
  no_error 6,435，**无 stringing**。场景 = Prusa i3 + **红色立方件** + 工业相机（域差异极大）。
- ⛔ 排除：3D-ADAM（任务不匹配）；Garuda/JITK 2026 的 99.68%（高到不可信）；
  **DC-07 Kaggle `padraigvalenti/...`**（83MB CC0 已下但**不可用**：网络爬取含图库/渲染图 + 仅 fail/success 二分类）。
- 🔴 **知识库缺口**：kb_v2 有 8 个 img_classes 标签，DC-02R 只覆盖 3+2 近似，
  `Blobs/Zits`/`Z Seam`/`Overhang` **三标签完全无图像** → DC-04 补不上，只能自采。
- DC-02R **无材料/参数/EXIF 维度**（文件名全是时间戳）→ **「按材料/参数分层验证」无法设计，别再提**。
- 去泄漏：前缀 GT（召回 100%）∪ DINOv2 余弦≥0.995 → **1,749 group**。SSRN 7395570 验证
  **pHash仅召回 3.3% 真实同病灶组** → 不可只用。**group-aware 为主口径**。

## 关键文件

`src/vision/guards.py`（**13 道防线**，`--self-test` 全绿）｜`weights.py`（健壮权重加载）｜
`dedup.py`（去泄漏）｜`train.py`（L1/L2/L3，读 D2 折号）｜`infer.py`+`pipeline.py`+`kb_mapping.py`｜
`src/skill_diagnose.py`+`agent_graph.py`（P0）｜`skill_vision_diagnose.py`+`agent_vision.py`（接线）｜
`scripts/fetch_zenodo_dc04.py`（6IP×3 并发下载器）

## 坑（别重犯）

**网络** 🔴 **并发不是越高越好**：18 并发短时 500 KB/s 但触发限流（掉到 3~7 KB/s）；
**6 并发稳态 746 KB/s 反而最快**（12× 提速，85h→11h）。限流两形态：降速(rc=0)/连不上(rc=7)。
Zenodo 需 `curl --resolve zenodo.org:443:<IP>` 绕 DNS 污染，**6 个可达 IPv4 全支持 Range 206**、可叠加。
❌`huggingface.co`超时/`modelscope.cn` status=000（不必再试）/`mirrors.aliyun.com` 不通；
✅ 唯一权重源 = **`hf-mirror.com`**。🔴 Kaggle `totalBytes` 元数据**不可作完整性基准**
（差 13.6MB 是 Kaggle 侧误差，按它补下会被拒 InvalidRange）→ **判据 = zip 条目数 + CRC + 各类张数**。
⚠️ `nohup &` 会被 Bash 会话回收 → 用 `run_in_background` + `python -u`。

**脚本** 🔴 **别用 `tmp.unlink()` 清残片**：删除被安全策略拦截 → PermissionError → 崩在 63%、
留 1098 孤儿文件。**不需要 unlink**（`curl -o` 本身就覆盖截断，实测 5MB+1MB→1,048,576 B）。
🔴 **限流时 rate=0 会 ZeroDivisionError 崩掉，且把「正在被限流」现场信息一起带走** → `rate>0` 才算eta，
否则显示「限流」**且不中断**。🔴 **别拿短时峰值当结论**（本项目已犯 3 次）。
⚠️ **别用 `except: pass` 掩盖导入失败**（`scripts/` 下 `sys.path[0]` 是 `scripts/` → 白排查半小时）→
用 `importlib.util.spec_from_file_location` + 显式 `sys.modules`。
⚠️ 别凭记忆写 `old_string`；`global` 写在 `parse_args()` **之后 → SyntaxError**；
FAISS `search` 返 `(scores, indices)`；属性是 `r.vector_index` 不是 `faiss_index`。
⚠️ git 用 `-c user.name=avoid137 -c user.email=1303572692@qq.com`（不动全局）。

**数据/权重** 🔴 **pHash/dHash 彻底降级为只读诊断**（误合并率 94.3%，闭包产出 19 个跨类污染 group），别回退。
🔴 **禁用 `timm.create_model(pretrained=True)`**；`load_pretrained_into` 校验注入张量数，0 个即抛。
`_read_state_dict` 用**魔数嗅探**（HF blob 无扩展名但内容是 safetensors，按 .pth 走 torch.load 报
`UnpicklingError: 176`）。`header too small` ≠ 网络失败（Windows 符号链接失败）。
`EXPECTED_BYTES` 全为实测：B0=21,355,344、DINOv2=**346,334,872**。
⚠️ `data/folds.json` path 是**绝对路径**（数据在 D 盘），`relative_to(ROOT)` 会 ValueError。
⚠️ **`data/mapping.md` 已失效**（基于弃用的 8 类体系，8 个 KB ID 不在 kb_v2 里）。

## 方法论纪律（本项目最值钱的部分）

**① 字面对得上≠ 现象对得上。** 两条映射都按字面语义推断、看着完全合理，**看图后全错**：
Cracking 20/20 是层向贯穿+撕裂纤维+层间错位 → 实为**层间结合失效 Delamination** → 主映射 **KB009**；
Off_platform 20/20 画面**无成形件**、材料堆成乱丝 → 是 **Spaghetti 挤出失败**，而
**KB101 标题就是 `Spaghetti (Print Detached from Bed)`** → 不能因「标题像别的东西」降级为易混淆。
⚠️ KB 标题可能含**旧版的、正当的别名**。⚠️ DC-02R 构图 = 整台打印机 + 底下一小块件，
**只看缩略图会误判，必须放大。**

**② 判「模型是否走捷径」不能只看「该线索单独能到多少分」，必须主动移除它看分数掉多少。**
19 维低级像素统计就能拿 71.5% macro-F1、各类背景色两两分明 → 我据此**推断**「99% 反映的是识别拍摄批次」
→ **被干预实验推翻**（去背景 0.00pp、去色+去背景仅 −0.33pp）。低阶捷径客观存在但线性探针并不依赖它。
延伸：遮挡指标**必须用连续值**（概率相对降幅），0/1 翻转在 5 分类里只有两档，全是 0 只说明「没翻转」。

## 回迁

`HANDOFF_P1.md` → `python scripts/p1_check_env.py`（应 5/5 绿）→ 本文件 → `exp/p1_vision_result.md`。