# 3D 打印故障诊断 Agent · 项目记忆（仓库自动加载）

> 本文件随仓库 clone 到新机器；把仓库根目录作为 WorkBuddy 工作区打开时自动注入。
> 完整接手文档始终是仓库根的 **HANDOFF_P1.md**（权威、面向 Agent 与人都可读）。

## 当前主线：P1（视觉分类，RTX 4060 机器）

- **P0 已交付且严格验证到顶**（别再碰）：strict SOTA **54.37%**(87/160, qwen-turbo 判官，判官间一致率 74%→不确定度约 **±7pp**)、lenient 69.37%、拒答误拒 0。检索侧（top-k=10 / rerank / query 改写 / 索引变体 8 个）与生成层（v2.3 净负 / v2.8 +2.5pp 但 23 回归 / v2.8b 等价）全部证伪或不过零回归安全门。结论见 `exp/generation_layer_campaign_conclusion.md`。
- **P1 任务**：5 类缺陷图像分类（Kaggle `wengmhu` / DC-02R，MIT 许可，1,912 张）+ 视觉→知识库映射层。数据集尚未下载。
- **P1 只动 `src/vision/`**，不要改 P0 检索/生成层（重做必回归）。

## 新机器接手顺序（必做）

1. 开训前先加载 skill **`kaggle-vision-pipeline-guards`**（五道防线：防"数据没喂进去但 loss 照降"这类不报错的错误）。
2. 环境：建 `.venv` → `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`（4060 必须 cu124，装 CPU 版不报错但慢 20×）→ `python scripts/p1_check_env.py` 全绿（含 GPU 实跑冒烟）。
3. 数据：`python scripts/p1_fetch_dataset.py --dir D:/datasets/fdm-defect`，校验 `data/p1_dataset_manifest.json` 的 `ok:true`。类别张数 538/472/447/364/91=1,912；⚠️ Kaggle README 的 Cracking=427 是笔误，实为 472。
4. 训练：EfficientNet-B0 / MobileNetV3（timm，ImageNet 预训练）+ 分层 5 折（`seed=42`）+ **macro-F1 均值±标准差**（Off_platform 仅 91 张，禁报单折）。

## 纪律红线

- 原始图像 / 权重（`*.pth` `*.pt`）不进 git（`.gitignore` 已排除 `data/raw` `data/datasets`）。
- 数字做完才填；对外口径标「qwen-turbo 判官，不确定度约 ±7pp」；不把论文 Swin-T 87% 说成自己的。
- 跨材料口径一致；类别映射 3 类同名直接映射，Cracking / Off_platform 须逐条核对 KB 后定并标注依据。
- 只采用许可明确的源（DC-02R = MIT）；Unknown 不入库。数据集台账 = `data/dataset_cards.md`。

## 回迁

同仓开发直接 push；P1 产物放 `src/vision/`、报告 `exp/p1_vision_result.md`、折划分 json 与 `data/p1_dataset_manifest.json` 提交；回本机 `git pull` 同步。
