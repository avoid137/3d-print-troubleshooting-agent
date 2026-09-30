# v2.8b 混合（v2.8 草稿 + append-only 回填保底）—— 实验结果

## 总览（in-scope 共 160 题）

- 回填基线 correct 率： **54.37%** ({'wrong': 37, 'correct': 87, 'partial': 36})
- v2.8 单独 correct 率： **56.87%** ({'wrong': 18, 'correct': 91, 'partial': 51})
- **v2.8b correct 率： 56.87% ({'wrong': 18, 'correct': 91, 'partial': 51})  = v2.8 完全等价**
- Δ vs 回填： **+2.50 pp** ｜ Δ vs v2.8： +0.00 pp
- 回归 vs 回填（须=0）： **23** ← 安全门未过（与 v2.8 相同 23 道）
- 救回 vs 回填： 27
- 回归 vs v2.8（回填须零新增）： **0** ✅ ｜ 救回 vs v2.8： 0
- lenient：回填 69.37% ／ v2.8 85.62% ／ v2.8b 85.62%

## 核心结论
- v2.8b **严格等价于 v2.8**（91/160，23 回归原封不动）。
- 回填保底**零新增回归但也零回收**：v2.8 的 skipped 是**显式拒绝**根因（答案写"未采纳 KBxxx"），
  回填只补"默默漏掉"的相关根因、不推翻主动拒绝 → 混合方案救不了 skip 回归。
- 故 v2.8 系列（含混合）**无法在零回归前提下超过回填 SOTA 54.37%**。

## 类别拆解（回填 / v2.8 / v2.8b）

| 类别 | n | 回填正确 | v2.8正确 | v2.8b正确 | 回填率 | v2.8率 | v2.8b率 |
|---|---|---|---|---|---|---|---|
| param_compensation | 40 | 21 | 22 | 22 | 52.50% | 55.00% | 55.00% |
| ambiguous | 40 | 23 | 25 | 25 | 57.50% | 62.50% | 62.50% |
| single_defect | 50 | 30 | 27 | 27 | 60.00% | 54.00% | 54.00% |
| multi_defect | 30 | 13 | 17 | 17 | 43.33% | 56.67% | 56.67% |

## 回归 vs 回填（23 道，与 v2.8 完全相同）

- `eval_008` [single_defect] 回填=correct → v2.8b=partial｜bridges sag and string
- `eval_025` [multi_defect] 回填=correct → v2.8b=partial｜ringing artifacts on print sides and overhang qual
- `eval_026` [single_defect] 回填=correct → v2.8b=partial｜surface has blobs and imperfections
- `eval_037` [single_defect] 回填=correct → v2.8b=partial｜filament not coming out of the nozzle
- `eval_044` [multi_defect] 回填=correct → v2.8b=partial｜print looks over extruded and droopy and printer s
- `eval_045` [single_defect] 回填=correct → v2.8b=partial｜print looks over extruded and droopy
- `eval_062` [param_compensation] 回填=correct → v2.8b=partial｜infill density too low weak print
- `eval_066` [param_compensation] 回填=correct → v2.8b=wrong｜retraction prime amount
- `eval_068` [param_compensation] 回填=correct → v2.8b=wrong｜cooling fan for overhang quality
- `eval_104` [single_defect] 回填=correct → v2.8b=partial｜blobs and zits on the print surface
- `eval_119` [ambiguous] 回填=correct → v2.8b=partial｜cooling ok but print looks melted
- `eval_121` [ambiguous] 回填=correct → v2.8b=wrong｜not over extrusion but print droopy
- `eval_122` [ambiguous] 回填=correct → v2.8b=wrong｜no clog but spaghetti mess
- `eval_123` [single_defect] 回填=correct → v2.8b=partial｜print splitting along layer lines
- `eval_142` [single_defect] 回填=correct → v2.8b=partial｜blocked nozzle not extruding
- `eval_159` [single_defect] 回填=correct → v2.8b=partial｜layers coming apart
- `eval_164` [ambiguous] 回填=correct → v2.8b=partial｜no layer shift but print leans
- `eval_168` [single_defect] 回填=correct → v2.8b=partial｜over extrusion making blobby prints
- `eval_176` [param_compensation] 回填=correct → v2.8b=partial｜wipe distance for oozing
- `eval_184` [single_defect] 回填=correct → v2.8b=partial｜my print is stringy and droopy over extrusion
- `eval_193` [param_compensation] 回填=correct → v2.8b=partial｜extrusion multiplier calibration
- `eval_194` [param_compensation] 回填=correct → v2.8b=partial｜travel speed causing blobs
- `eval_199` [single_defect] 回填=correct → v2.8b=partial｜delamination between layers

## 救回 vs 回填（27 道，来自 v2.8 的 27 道增益，回填保底未增减）

- `eval_007` [ambiguous] 回填=partial → v2.8b=correct｜not stringing but web-like strands
- `eval_014` [single_defect] 回填=wrong → v2.8b=correct｜how to unclog a 3D printer nozzle
- `eval_020` [ambiguous] 回填=wrong → v2.8b=correct｜nozzle is not clogged but filament grinds
- `eval_024` [param_compensation] 回填=partial → v2.8b=correct｜brim width for bed adhesion
- `eval_027` [ambiguous] 回填=wrong → v2.8b=correct｜no ringing but ghosting on sides
- `eval_043` [single_defect] 回填=partial → v2.8b=correct｜ripples and echoes on surface
- `eval_072` [multi_defect] 回填=partial → v2.8b=correct｜layer misalignment on tall prints and also extrude
- `eval_074` [multi_defect] 回填=partial → v2.8b=correct｜grinding filament in the extruder and the part is 
- `eval_075` [param_compensation] 回填=partial → v2.8b=correct｜bridging fan speed
- `eval_083` [multi_defect] 回填=partial → v2.8b=correct｜my print has stringing and oozing between parts an
- `eval_086` [ambiguous] 回填=partial → v2.8b=correct｜cooling is fine but overhangs droop
- `eval_090` [single_defect] 回填=wrong → v2.8b=correct｜layer misalignment on tall prints
- `eval_097` [multi_defect] 回填=partial → v2.8b=correct｜there are thin plastic strings on my print and ext
- `eval_105` [single_defect] 回填=partial → v2.8b=correct｜print not sticking to the bed
- `eval_107` [param_compensation] 回填=partial → v2.8b=correct｜infill pattern strength
- `eval_112` [single_defect] 回填=partial → v2.8b=correct｜edges bending upwards during printing
- `eval_116` [ambiguous] 回填=wrong → v2.8b=correct｜the print sticks but first layer is messy
- `eval_126` [single_defect] 回填=partial → v2.8b=correct｜first layer won't adhere
- `eval_131` [param_compensation] 回填=partial → v2.8b=correct｜bed leveling for first layer
- `eval_132` [multi_defect] 回填=partial → v2.8b=correct｜scars on the top surface of print and too much fil
- `eval_140` [single_defect] 回填=partial → v2.8b=correct｜print lifts off the build plate
- `eval_154` [param_compensation] 回填=partial → v2.8b=correct｜bed temperature for PLA warping
- `eval_156` [ambiguous] 回填=partial → v2.8b=correct｜first layer ok but top has gaps
- `eval_178` [single_defect] 回填=wrong → v2.8b=correct｜extruding too much plastic
- `eval_179` [param_compensation] 回填=partial → v2.8b=correct｜first layer height too high not sticking
- `eval_189` [multi_defect] 回填=partial → v2.8b=correct｜extrusion stopped mid print and print turned into 
- `eval_191` [param_compensation] 回填=wrong → v2.8b=correct｜backlash compensation dimensional