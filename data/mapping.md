# 图像缺陷类别 ↔ 知识库条目 映射表 (v1)

> ⚠️ **2026-09-28 晚注：本表基于已弃用的 Minds-R-Lab 8 类体系。**
> 该数据集经核查**不可获取**（仓库无数据文件、无 LICENSE、无 release），
> P1 已换源为 Kaggle `wengmhu/fdm-3d-printing-defect-dataset`（MIT，1,912 张 / 5 类）。
> **本表待按 5 类体系重做**；重做前仅供了解「命名体系差异」的维度参考，**不可作为 P1 映射层依据**。
> 新 5 类 ↔ 知识库的初步对齐结论见 `00_项目准备总览.md` §三。

> 生成日期：2026-09-28 ｜ 上游：Minds-R-Lab 3D-Printing-Dataset(8类+Okay) ↔ 3D Printing Troubleshooting Guide(75条)
> ⚠️ 两侧**命名体系不同**（图像用视觉术语，知识库用现象描述），本表为人工映射基准。
> `match` 字段：`direct`=名称可直接对应 / `semantic`=语义对应需人工确认 / `weak`=覆盖薄弱需补 / `todo`=待补充

## 1. 覆盖总览

| 图像类别 | 中文 | 知识条目数 | 根因数 | 状态 |
|---|---|---|---|---|
| Spaghetti | 空中乱丝 | 1 | 2 | ⚠️ 薄弱(需补) |
| Stringing | 拉丝/渗料 | 2 | 15 | ✅ |
| Blobs/Zits | 表面疙瘩 | 2 | 8 | ✅ |
| Warping | 翘曲 | 4 | 15 | ✅ |
| Z Seam | 层接缝 | 2 | 7 | ✅ |
| Layer Shifting | 层错位 | 5 | 25 | ✅ |
| Delamination | 层间分离 | 3 | 10 | ✅ |
| Overhang | 悬垂/桥接 | 3 | 14 | ✅ |

## 2. 逐类映射明细

### Spaghetti（空中乱丝）

- `KB059` Print Looks Stringy and Droopy (Over-Extrusion)  —— 根因 2 条 ｜ match=`weak` ｜ FDM=True

### Stringing（拉丝/渗料）

- `KB006` Stringing or Oozing  —— 根因 12 条 ｜ match=`direct` ｜ FDM=True
- `KB063` Web-like Strings Cover the Print (Stringing)  —— 根因 3 条 ｜ match=`semantic` ｜ FDM=True

### Blobs/Zits（表面疙瘩）

- `KB014` Blobs and Zits  —— 根因 4 条 ｜ match=`direct` ｜ FDM=True
- `KB062` Pits and Hollows in Top Layer (Pillowing)  —— 根因 4 条 ｜ match=`semantic` ｜ FDM=True

### Warping（翘曲）

- `KB016` Curling or Rough Corners  —— 根因 0 条 ｜ match=`direct` ｜ FDM=True
- `KB024` Warping  —— 根因 5 条 ｜ match=`direct` ｜ FDM=True
- `KB044` Print Bows Out at Bottom (Elephant's Foot)  —— 根因 4 条 ｜ match=`direct` ｜ FDM=True
- `KB045` Print Edges are Bending (Warping)  —— 根因 6 条 ｜ match=`semantic` ｜ FDM=True

### Z Seam（层接缝）

- `KB017` Scars on Top Surface  —— 根因 2 条 ｜ match=`semantic` ｜ FDM=True
- `KB058` Diagonal Scars on Print  —— 根因 5 条 ｜ match=`semantic` ｜ FDM=True

### Layer Shifting（层错位）

- `KB008` Layer Shifting  —— 根因 2 条 ｜ match=`direct` ｜ FDM=True
- `KB029` Layer Shifting  —— 根因 5 条 ｜ match=`direct` ｜ FDM=True
- `KB050` Layers Don't Line Up Well  —— 根因 6 条 ｜ match=`semantic` ｜ FDM=True
- `KB052` Print Leans When it Shouldn't  —— 根因 4 条 ｜ match=`semantic` ｜ FDM=True
- `KB065` Print Offset in Some Places  —— 根因 8 条 ｜ match=`semantic` ｜ FDM=True

### Delamination（层间分离）

- `KB009` Layer Separation and Splitting  —— 根因 5 条 ｜ match=`direct` ｜ FDM=True
- `KB049` Cracks Have Appeared in Tall Objects  —— 根因 2 条 ｜ match=`direct` ｜ FDM=True
- `KB073` Layers Have Separated (Delamination)  —— 根因 3 条 ｜ match=`semantic` ｜ FDM=True

### Overhang（悬垂/桥接）

- `KB027` Poor Bridging  —— 根因 4 条 ｜ match=`direct` ｜ FDM=True
- `KB053` Overhangs are Messy  —— 根因 5 条 ｜ match=`semantic` ｜ FDM=True
- `KB066` Bridges are Messy  —— 根因 5 条 ｜ match=`semantic` ｜ FDM=True

## 3. 待补清单（P0 必做）

| # | 项 | 原因 | 动作 |
|---|---|---|---|
| 1 | **Spaghetti 专属条目** | 图像有该类，知识库仅 1 条间接覆盖；且 Spaghetti(空中挤出) 与 Stringing(回抽不足) **成因不同**，硬映射会答错 | 从 All3DP / Simplify3D 补 2–3 条 |
| 2 | 非 FDM 条目剔除 | 3 条属光固化/激光，与 FDM 图像数据集不匹配 | 见下 §4 |
| 3 | 重复/同义条目合并 | 1 组完全重名 + 大量同义表述 | 合并后产出"75→N"数字 |

## 4. 非 FDM 条目（剔除）

- `KB030` Print doesn't appear / Resin does not solidify
- `KB068` Resin too Cold
- `KB070` Underpowered Laser

## 5. 知识库有、图像数据集无对应类（53 条）

> **设计含义（重要）**：知识库覆盖面**远大于**图像数据集。
> "喷嘴堵塞""首层不粘"这类故障在图像上不明显，**必须靠传感器数据 + 用户描述补充**。
> **这正是 Multi-Agent 多源融合的立项理由** —— 单一视觉通道不够。

共 50 条（已排除非 FDM）。示例：

- `KB001` Not Extruding at Start of Print（根因 4）
- `KB002` Not Sticking to the Bed（根因 6）
- `KB003` Under-Extrusion（根因 9）
- `KB004` Over-Extrusion（根因 0）
- `KB005` Gaps in Top Layers（根因 3）
- `KB007` Overheating（根因 4）
- `KB010` Grinding Filament（根因 4）
- `KB011` Clogged Extruder（根因 3）
- `KB012` Stops Extruding Mid Print（根因 4）
- `KB013` Weak Infill（根因 3）
- `KB015` Gaps Between Infill and Outline（根因 2）
- `KB018` Gaps in Floor Corners（根因 3）
- `KB019` Lines on the Side of Print（根因 3）
- `KB020` Vibrations and Ringing（根因 3）
- `KB021` Gaps in Thin Walls（根因 2）
- `KB022` Small Features Not Printed（根因 3）
- `KB023` Inconsistent Extrusion（根因 6）
- `KB025` Poor Surface Above Supports（根因 5）
- `KB026` Dimensional Accuracy（根因 4）
- `KB028` First Layer Issues（根因 11）
- ... 另 30 条