"""构建 200 条评测集。

五类分布：
  单缺陷      25%  50 条  — 单一故障描述，gold = [该 kb_id]
  多缺陷      15%  30 条  — 两个故障同时出现，gold = [kb_id_a, kb_id_b]
  需参数补偿  20%  40 条  — 提到具体参数（回抽/温度/速度等），gold = [相关 kb_id]
  库外负例    20%  40 条  — 知识库不覆盖的问题，gold = []（应拒答）
  诱导歧义    20%  40 条  — 含干扰词但指向某一故障，gold = [目标 kb_id]

gold 字段存 kb_id 列表（可能多个）。负例 gold = []。
"""
import json
import random
from config import KB_V2_PATH, EVAL_SET_PATH

random.seed(42)

# 手写的查询模板：问题描述 → 该故障的多种问法
# key: kb_id, value: list of query strings (paraphrases)
SINGLE_DEFECT_QUERIES = {
    "KB006": [
        "my print has stringing and oozing between parts",
        "there are thin plastic strings on my print",
        "hairy prints with whiskers of filament",
        "oozing plastic while the nozzle moves",
        "how to fix stringing on my 3D print",
    ],
    "KB024": [
        "the corners of my print are curling up",
        "print warping and lifting off the bed",
        "edges bending upwards during printing",
        "the part is warping at the corners",
        "how to prevent warping",
    ],
    "KB008": [
        "layers are shifting in my print",
        "the print layers don't line up",
        "layer misalignment on tall prints",
        "print looks offset in some layers",
        "layers shifted horizontally",
    ],
    "KB011": [
        "nozzle is clogged filament won't come out",
        "extruder is clogged",
        "blocked nozzle not extruding",
        "filament not coming out of the nozzle",
        "how to unclog a 3D printer nozzle",
    ],
    "KB003": [
        "print looks under extruded",
        "gaps in the print walls",
        "weak thin layers from under extrusion",
        "not enough plastic being extruded",
        "under extrusion causing gaps",
    ],
    "KB004": [
        "print looks over extruded and droopy",
        "too much filament coming out",
        "over extrusion making blobby prints",
        "extruding too much plastic",
        "my print is stringy and droopy over extrusion",
    ],
    "KB002": [
        "print not sticking to the bed",
        "first layer won't adhere",
        "print lifts off the build plate",
        "part doesn't stick to print bed",
        "how to make print stick to bed",
    ],
    "KB014": [
        "blobs and zits on the print surface",
        "small bumps on my 3D print",
        "surface has blobs and imperfections",
        "zits appearing on the print",
    ],
    "KB009": [
        "layers separating and splitting",
        "delamination between layers",
        "layers coming apart",
        "cracks between printed layers",
        "print splitting along layer lines",
    ],
    "KB027": [
        "bridging looks messy and droopy",
        "poor bridging on my print",
        "bridges sag and string",
        "failed bridging over gaps",
    ],
    "KB020": [
        "vibrations and ringing on the print",
        "ripples and echoes on surface",
        "ringing artifacts on print sides",
        "ghosting on the print surface",
    ],
    "KB007": [
        "print overheating and deforming",
        "print looks melted",
        "overheating causing layer deformation",
        "heat deformation on tall prints",
    ],
    "KB101": [
        "spaghetti mess filament tangled in air",
        "print turned into spaghetti",
        "filament extruded in the air tangled",
        "print became a mess of plastic strands",
    ],
    "KB053": [
        "overhangs are messy and drooping",
        "overhang quality is poor",
        "droopy overhangs on print",
    ],
    "KB017": [
        "scars on the top surface of print",
        "z seam marks on print",
        "visible seam on print surface",
        "diagonal scars on my print",
    ],
    "KB028": [
        "first layer issues on my print",
        "first layer looks messy",
        "problems with the first layer",
    ],
    "KB010": [
        "extruder is grinding the filament",
        "filament stripped by drive gear",
        "grinding filament in the extruder",
    ],
    "KB001": [
        "not extruding at the start of print",
        "no filament comes out at beginning",
        "extruder not primed at print start",
    ],
    "KB040": [
        "extrusion stopped mid print",
        "printer stopped extruding halfway",
        "filament stops coming out during print",
    ],
    "KB026": [
        "dimensional accuracy issues",
        "print dimensions are wrong",
        "part is not the right size",
    ],
}

# 参数补偿类查询：提到具体打印参数
PARAM_QUERIES = [
    ("what retraction distance should I use for stringing", ["KB006"]),
    ("retraction speed too high causing grinding", ["KB010", "KB006"]),
    ("bed temperature for PLA warping", ["KB024", "KB002"]),
    ("nozzle temperature too low under extrusion", ["KB003"]),
    ("print speed too fast causing layer shift", ["KB008"]),
    ("cooling fan settings for overhangs", ["KB053", "KB027"]),
    ("infill density too low weak print", ["KB013"]),
    ("first layer height too high not sticking", ["KB002", "KB028"]),
    ("flow rate too high over extrusion", ["KB004"]),
    ("z offset too close to bed", ["KB035"]),
    ("acceleration too high ringing", ["KB020"]),
    ("retraction when crossing perimeters stringing", ["KB006"]),
    ("hotend temp for PETG oozing", ["KB006"]),
    ("build plate temperature for ABS", ["KB024"]),
    ("print speed for bridging", ["KB027"]),
    ("fan speed for first layer adhesion", ["KB002"]),
    ("layer height affecting detail", ["KB022"]),
    ("extrusion multiplier calibration", ["KB003", "KB004"]),
    ("travel speed causing blobs", ["KB014"]),
    ("z seam alignment settings", ["KB017"]),
    ("how much retraction for bowden extruder", ["KB006"]),
    ("temperature tower calibration", ["KB003", "KB004"]),
    ("bed leveling for first layer", ["KB028", "KB002"]),
    ("cooling fan for overhang quality", ["KB053"]),
    ("nozzle diameter change under extrusion", ["KB003"]),
    ("filament diameter settings", ["KB003"]),
    ("support density for surface finish", ["KB025"]),
    ("brim width for bed adhesion", ["KB002"]),
    ("elephant foot compensation", ["KB044"]),
    ("ironing settings top surface", ["KB005"]),
    ("coasting settings reduce blobs", ["KB014"]),
    ("wipe distance for oozing", ["KB006"]),
    ("minimal layer time cooling", ["KB007"]),
    ("bridging fan speed", ["KB027"]),
    ("infill pattern strength", ["KB013"]),
    ("wall count for strength", ["KB013"]),
    ("nozzle wipe for stringing", ["KB006"]),
    ("retraction prime amount", ["KB006"]),
    ("jerk settings layer shift", ["KB008"]),
    ("backlash compensation dimensional", ["KB026"]),
]

# 库外负例：知识库不覆盖的问题（应拒答）
OUT_OF_SCOPE_QUERIES = [
    "how to design a 3D model in CAD",
    "what is the best 3D printer to buy",
    "how to scan objects for 3D printing",
    "slicer settings for resin printing",
    "how to paint 3D printed parts",
    "best filament storage solutions",
    "how to model organic shapes",
    "3D printer firmware setup marlin",
    "how to make money with 3D printing",
    "what is SLA printing",
    "how to calibrate a laser engraver",
    "best CAD software for beginners",
    "how to recycle used filament",
    "3D printing business ideas",
    "how to connect printer to octoprint",
    "resin vat cleaning",
    "DLP vs SLA resin printing",
    "how to post process resin prints",
    "best 3D printing youtube channels",
    "how to level a resin printer",
    "what is FDM vs SLA",
    "how to choose nozzle material",
    "3D printing medical implants",
    "how to print flexible TPU",
    "best enclosure for ABS printing",
    "how to wire a 3D printer",
    "3D printer mainboard upgrade",
    "how to use supports in slicer",
    "best slicer for 3D printing",
    "how to clean PETG prints",
    "3D printing tolerances guide",
    "how to print with nylon",
    "best build plate surface",
    "how to change nozzle",
    "3D printing speed records",
    "how to print multi color",
    "best infill for strength",
    "how to use a 3D pen",
    "3D printing food safe",
    "how to print with carbon fiber",
]

# 诱导歧义：含干扰词但指向某一故障
AMBIGUOUS_QUERIES = [
    ("my print has strings but also the corners lift", ["KB006", "KB024"]),
    ("nozzle is not clogged but filament grinds", ["KB010"]),
    ("the print sticks but first layer is messy", ["KB028"]),
    ("not overheating but layers separate", ["KB009"]),
    ("no stringing but blobs on surface", ["KB014"]),
    ("bed is level but print won't stick", ["KB002"]),
    ("retraction is fine but still oozing", ["KB006"]),
    ("not warping but dimensional inaccurate", ["KB026"]),
    ("no layer shift but print leans", ["KB052"]),
    ("cooling is fine but overhangs droop", ["KB053"]),
    ("not under extruded but walls are weak", ["KB013"]),
    ("no ringing but surface ripples", ["KB019"]),
    ("first layer ok but top has gaps", ["KB005"]),
    ("no clog but stops extruding mid print", ["KB040"]),
    ("not stringing but hairy surface", ["KB063"]),
    ("bed temp correct but corners curl", ["KB016"]),
    ("not over extrusion but print droopy", ["KB059"]),
    ("no delamination but cracks in tall print", ["KB049"]),
    ("retraction good but blobs at seams", ["KB017"]),
    ("not clogging but extrusion inconsistent", ["KB023"]),
    ("print sticks but elephant foot at bottom", ["KB044"]),
    ("no warp but edges bending", ["KB045"]),
    ("not stringing but web-like strands", ["KB063"]),
    ("cooling ok but print looks melted", ["KB061"]),
    ("no shift but layers don't line up", ["KB050"]),
    ("not bridging issue but messy bridges", ["KB066"]),
    ("no clog but spaghetti mess", ["KB101"]),
    ("bed adhesion good but print detached", ["KB101"]),
    ("not over extruded but stringy droopy", ["KB059"]),
    ("retraction fine but whiskers on print", ["KB006"]),
    ("no ringing but ghosting on sides", ["KB057"]),
    ("not under extruded but gaps in walls", ["KB021"]),
    ("first layer fine but infill visible outside", ["KB048"]),
    ("no clog but filament stripped", ["KB039"]),
    ("not warping but print bows at bottom", ["KB044"]),
    ("no stringing but oozing between moves", ["KB006"]),
    ("not dimensional issue but parts too small", ["KB064"]),
    ("no clog but filament snaps", ["KB038"]),
    ("not overheating but ripples on surface", ["KB057"]),
    ("no adhesion issue but print stuck to bed", ["KB067"]),
]


def build_eval_set():
    eval_set = []

    # 1. 单缺陷 50 条
    single = []
    for kb_id, queries in SINGLE_DEFECT_QUERIES.items():
        for q in queries:
            if len(single) >= 50:
                break
            single.append({
                "id": f"single_{len(single)+1:03d}",
                "query": q,
                "category": "single_defect",
                "gold": [kb_id],
            })
        if len(single) >= 50:
            break
    eval_set.extend(single)

    # 2. 多缺陷 30 条：两两组合
    defect_ids = list(SINGLE_DEFECT_QUERIES.keys())
    multi = []
    random.shuffle(defect_ids)
    for i in range(0, min(60, len(defect_ids) - 1), 2):
        a, b = defect_ids[i], defect_ids[i + 1]
        qa = random.choice(SINGLE_DEFECT_QUERIES[a])
        qb = random.choice(SINGLE_DEFECT_QUERIES[b])
        combined = f"{qa} and also {qb}"
        multi.append({
            "id": f"multi_{len(multi)+1:03d}",
            "query": combined,
            "category": "multi_defect",
            "gold": [a, b],
        })
    # 补足到 30
    while len(multi) < 30:
        a, b = random.sample(defect_ids, 2)
        qa = random.choice(SINGLE_DEFECT_QUERIES[a])
        qb = random.choice(SINGLE_DEFECT_QUERIES[b])
        multi.append({
            "id": f"multi_{len(multi)+1:03d}",
            "query": f"{qa} and {qb}",
            "category": "multi_defect",
            "gold": [a, b],
        })
    eval_set.extend(multi[:30])

    # 3. 需参数补偿 40 条
    for q, gold in PARAM_QUERIES:
        eval_set.append({
            "id": f"param_{len(eval_set)+1:03d}",
            "query": q,
            "category": "param_compensation",
            "gold": gold,
        })

    # 4. 库外负例 40 条
    for q in OUT_OF_SCOPE_QUERIES:
        eval_set.append({
            "id": f"neg_{len(eval_set)+1:03d}",
            "query": q,
            "category": "out_of_scope",
            "gold": [],
        })

    # 5. 诱导歧义 40 条
    for q, gold in AMBIGUOUS_QUERIES:
        eval_set.append({
            "id": f"amb_{len(eval_set)+1:03d}",
            "query": q,
            "category": "ambiguous",
            "gold": gold,
        })

    random.shuffle(eval_set)
    # 重新编号
    for i, e in enumerate(eval_set, 1):
        e["id"] = f"eval_{i:03d}"

    # 统计
    from collections import Counter
    cats = Counter(e["category"] for e in eval_set)
    print(f"评测集总数: {len(eval_set)}")
    for cat, cnt in cats.items():
        print(f"  {cat}: {cnt}")

    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"已写入 {EVAL_SET_PATH}")


if __name__ == "__main__":
    build_eval_set()
