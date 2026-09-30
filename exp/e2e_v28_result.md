# v2.8 枚举式结构化穷尽 —— 实验结果

## 总览（in-scope 共 160 题）

- 回填基线 correct 率： **54.37%** ({'wrong': 37, 'correct': 87, 'partial': 36})
- v2.8 correct 率： **56.87%** ({'wrong': 18, 'correct': 91, 'partial': 51})
- 净 Δpp： **+2.50 pp**
- 救回数（回填非 correct → v28 correct）： **27**
- 回归数（回填 correct → v28 非 correct）： **23**
- lenient：回填 69.37% → v2.8 85.62%

## 目标集（回填后仍非 correct 的 73 题）

- 转 correct： **27** （救回）
- 仍非 correct： 46

转化矩阵（回填 → v2.8）：
-  partial → correct : 20
-  wrong → partial : 17
-  partial → partial : 15
-  wrong → wrong : 13
-  wrong → correct : 7
-  partial → wrong : 1

## 类别拆解

| 类别 | n | 回填正确 | v2.8正确 | 回填率 | v2.8率 | Δpp |
|---|---|---|---|---|---|---|
| param_compensation | 40 | 21 | 22 | 52.50% | 55.00% | +2.50 |
| ambiguous | 40 | 23 | 25 | 57.50% | 62.50% | +5.00 |
| single_defect | 50 | 30 | 27 | 60.00% | 54.00% | -6.00 |
| multi_defect | 30 | 13 | 17 | 43.33% | 56.67% | +13.34 |

## 救回明细（27 题）

- `eval_007` [ambiguous] 回填=partial → v2.8=correct｜not stringing but web-like strands
- `eval_014` [single_defect] 回填=wrong → v2.8=correct｜how to unclog a 3D printer nozzle
- `eval_020` [ambiguous] 回填=wrong → v2.8=correct｜nozzle is not clogged but filament grinds
- `eval_024` [param_compensation] 回填=partial → v2.8=correct｜brim width for bed adhesion
- `eval_027` [ambiguous] 回填=wrong → v2.8=correct｜no ringing but ghosting on sides
- `eval_043` [single_defect] 回填=partial → v2.8=correct｜ripples and echoes on surface
- `eval_072` [multi_defect] 回填=partial → v2.8=correct｜layer misalignment on tall prints and also extruder is grind
- `eval_074` [multi_defect] 回填=partial → v2.8=correct｜grinding filament in the extruder and the part is warping at
- `eval_075` [param_compensation] 回填=partial → v2.8=correct｜bridging fan speed
- `eval_083` [multi_defect] 回填=partial → v2.8=correct｜my print has stringing and oozing between parts and also blo
- `eval_086` [ambiguous] 回填=partial → v2.8=correct｜cooling is fine but overhangs droop
- `eval_090` [single_defect] 回填=wrong → v2.8=correct｜layer misalignment on tall prints
- `eval_097` [multi_defect] 回填=partial → v2.8=correct｜there are thin plastic strings on my print and extruder not 
- `eval_105` [single_defect] 回填=partial → v2.8=correct｜print not sticking to the bed
- `eval_107` [param_compensation] 回填=partial → v2.8=correct｜infill pattern strength
- `eval_112` [single_defect] 回填=partial → v2.8=correct｜edges bending upwards during printing
- `eval_116` [ambiguous] 回填=wrong → v2.8=correct｜the print sticks but first layer is messy
- `eval_126` [single_defect] 回填=partial → v2.8=correct｜first layer won't adhere
- `eval_131` [param_compensation] 回填=partial → v2.8=correct｜bed leveling for first layer
- `eval_132` [multi_defect] 回填=partial → v2.8=correct｜scars on the top surface of print and too much filament comi
- `eval_140` [single_defect] 回填=partial → v2.8=correct｜print lifts off the build plate
- `eval_154` [param_compensation] 回填=partial → v2.8=correct｜bed temperature for PLA warping
- `eval_156` [ambiguous] 回填=partial → v2.8=correct｜first layer ok but top has gaps
- `eval_178` [single_defect] 回填=wrong → v2.8=correct｜extruding too much plastic
- `eval_179` [param_compensation] 回填=partial → v2.8=correct｜first layer height too high not sticking
- `eval_189` [multi_defect] 回填=partial → v2.8=correct｜extrusion stopped mid print and print turned into spaghetti
- `eval_191` [param_compensation] 回填=wrong → v2.8=correct｜backlash compensation dimensional

## ⚠️ 回归明细（23 题，须为零）

- `eval_008` [single_defect] 回填=correct → v2.8=partial｜bridges sag and string
- `eval_025` [multi_defect] 回填=correct → v2.8=partial｜ringing artifacts on print sides and overhang quality is poo
- `eval_026` [single_defect] 回填=correct → v2.8=partial｜surface has blobs and imperfections
- `eval_037` [single_defect] 回填=correct → v2.8=partial｜filament not coming out of the nozzle
- `eval_044` [multi_defect] 回填=correct → v2.8=partial｜print looks over extruded and droopy and printer stopped ext
- `eval_045` [single_defect] 回填=correct → v2.8=partial｜print looks over extruded and droopy
- `eval_062` [param_compensation] 回填=correct → v2.8=partial｜infill density too low weak print
- `eval_066` [param_compensation] 回填=correct → v2.8=wrong｜retraction prime amount
- `eval_068` [param_compensation] 回填=correct → v2.8=wrong｜cooling fan for overhang quality
- `eval_104` [single_defect] 回填=correct → v2.8=partial｜blobs and zits on the print surface
- `eval_119` [ambiguous] 回填=correct → v2.8=partial｜cooling ok but print looks melted
- `eval_121` [ambiguous] 回填=correct → v2.8=wrong｜not over extrusion but print droopy
- `eval_122` [ambiguous] 回填=correct → v2.8=wrong｜no clog but spaghetti mess
- `eval_123` [single_defect] 回填=correct → v2.8=partial｜print splitting along layer lines
- `eval_142` [single_defect] 回填=correct → v2.8=partial｜blocked nozzle not extruding
- `eval_159` [single_defect] 回填=correct → v2.8=partial｜layers coming apart
- `eval_164` [ambiguous] 回填=correct → v2.8=partial｜no layer shift but print leans
- `eval_168` [single_defect] 回填=correct → v2.8=partial｜over extrusion making blobby prints
- `eval_176` [param_compensation] 回填=correct → v2.8=partial｜wipe distance for oozing
- `eval_184` [single_defect] 回填=correct → v2.8=partial｜my print is stringy and droopy over extrusion
- `eval_193` [param_compensation] 回填=correct → v2.8=partial｜extrusion multiplier calibration
- `eval_194` [param_compensation] 回填=correct → v2.8=partial｜travel speed causing blobs
- `eval_199` [single_defect] 回填=correct → v2.8=partial｜delamination between layers

## 目标集仍 non-correct 的样例（前 20，含 skipped 供归因）

- `eval_001` [param_compensation] 回填=wrong v2.8=wrong skip=['KB003', 'KB003', 'KB003', 'KB003', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB024', 'KB044', 'KB044', 'KB044', 'KB044', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB028', 'KB027', 'KB027', 'KB027', 'KB027', 'KB027', 'KB027', 'KB027', 'KB027']｜temperature tower calibration
- `eval_009` [multi_defect] 回填=partial v2.8=partial skip=['KB033']｜heat deformation on tall prints and hairy prints w
- `eval_010` [single_defect] 回填=wrong v2.8=wrong skip=['KB019', 'KB062', 'KB009']｜the print layers don't line up
- `eval_013` [ambiguous] 回填=wrong v2.8=partial skip=[]｜no warp but edges bending
- `eval_015` [param_compensation] 回填=wrong v2.8=wrong skip=['KB002', 'KB005', 'KB025']｜ironing settings top surface
- `eval_017` [param_compensation] 回填=partial v2.8=partial skip=['KB006']｜nozzle temperature too low under extrusion
- `eval_022` [single_defect] 回填=wrong v2.8=wrong skip=['KB051', 'KB051', 'KB051', 'KB051']｜weak thin layers from under extrusion
- `eval_031` [multi_defect] 回填=wrong v2.8=partial skip=['KB033', 'KB014', 'KB009', 'KB065']｜blobs and zits on the print surface and also crack
- `eval_032` [single_defect] 回填=wrong v2.8=wrong skip=['KB062', 'KB022']｜small bumps on my 3D print
- `eval_033` [ambiguous] 回填=partial v2.8=partial skip=['KB055']｜not under extruded but walls are weak
- `eval_034` [ambiguous] 回填=wrong v2.8=wrong skip=['KB004', 'KB004', 'KB053', 'KB053', 'KB053', 'KB053', 'KB053']｜not over extruded but stringy droopy
- `eval_036` [param_compensation] 回填=partial v2.8=partial skip=['KB014', 'KB014', 'KB014', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB006', 'KB010', 'KB010', 'KB010', 'KB010', 'KB010', 'KB010', 'KB010', 'KB010', 'KB010', 'KB023', 'KB023', 'KB023', 'KB023', 'KB023', 'KB023', 'KB023']｜how much retraction for bowden extruder
- `eval_049` [single_defect] 回填=partial v2.8=partial skip=['KB038', 'KB038', 'KB038', 'KB038', 'KB012', 'KB012', 'KB012', 'KB012', 'KB012', 'KB012', 'KB012', 'KB012', 'KB102', 'KB102', 'KB102', 'KB001', 'KB001', 'KB001', 'KB001', 'KB012']｜too much filament coming out
- `eval_050` [single_defect] 回填=wrong v2.8=partial skip=['KB019', 'KB019', 'KB019', 'KB005', 'KB005', 'KB005']｜layers are shifting in my print
- `eval_055` [multi_defect] 回填=wrong v2.8=partial skip=['KB046', 'KB053']｜first layer looks messy and also extrusion stopped
- `eval_061` [param_compensation] 回填=wrong v2.8=wrong skip=['KB008', 'KB026', 'KB006', 'KB051']｜z seam alignment settings
- `eval_069` [single_defect] 回填=wrong v2.8=partial skip=['KB007']｜not enough plastic being extruded
- `eval_070` [ambiguous] 回填=wrong v2.8=partial skip=['KB063', 'KB006', 'KB010', 'KB049']｜retraction good but blobs at seams
- `eval_077` [single_defect] 回填=partial v2.8=wrong skip=['KB055', 'KB025']｜print looks offset in some layers
- `eval_081` [param_compensation] 回填=wrong v2.8=partial skip=['KB003', 'KB003', 'KB003', 'KB003', 'KB003', 'KB023', 'KB023', 'KB023', 'KB023', 'KB023', 'KB023', 'KB031', 'KB031', 'KB004']｜flow rate too high over extrusion