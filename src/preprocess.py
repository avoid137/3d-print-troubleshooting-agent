"""数据清洗：剔除非 FDM + 同义合并 + 补充 Spaghetti 条目。

输入: data/kb_v1.json (75 条)
输出: data/kb_v2.json
"""
import json
from pathlib import Path
from config import KB_V1_PATH, KB_V2_PATH

# ---- 非 FDM 条目（光固化 / 激光），剔除 ----
NON_FDM_IDS = {"KB030", "KB068", "KB070"}

# ---- 同义合并组 ----
# 每组第一个 id 为主 id，其余并入；合并后 problem 名取主条目的 problem
MERGE_GROUPS = [
    ["KB008", "KB029"],                                # Layer Shifting（同名）
    ["KB011", "KB036"],                                # Clogged Extruder / Blocked Nozzle
    ["KB012", "KB040"],                                # Stops Extruding Mid Print
    ["KB002", "KB041", "KB071"],                       # Not Sticking to Bed
    ["KB015", "KB047"],                                # Gaps Between Infill and Outline
    ["KB020", "KB057"],                                # Vibrations and Ringing
    ["KB024", "KB045"],                                # Warping
    ["KB009", "KB073"],                                # Layer Separation / Delamination
    ["KB027", "KB066"],                                # Poor Bridging
    ["KB028", "KB043"],                                # First Layer Issues
    ["KB025", "KB054"],                                # Poor Surface Above Supports
    ["KB026", "KB064"],                                # Dimensional Accuracy
    ["KB003", "KB060"],                                # Under-Extrusion
    ["KB004", "KB059"],                                # Over-Extrusion
    ["KB022", "KB032", "KB056", "KB075"],              # Fine Details Not Printing
    ["KB010", "KB039"],                                # Grinding / Stripped Filament
]

# ---- Spaghetti 补充条目（2 条主因 + 1 条综合） ----
# 来源：All3DP / Simplify3D 关于 Spaghetti 的排障内容整理
# Spaghetti 与 Stringing 成因不同：Stringing 是回抽不足的细丝，
# Spaghetti 是打印彻底失败、耗材在空中乱缠，通常由打印脱落、堵头、耗材缠绕引起。
SPAGHETTI_ENTRIES = [
    {
        "kb_id": "KB101",
        "problem": "Spaghetti (Print Detached from Bed)",
        "is_fdm": True,
        "img_classes": ["Spaghetti"],
        "intro": (
            "Spaghetti occurs when the print detaches from the build plate mid-print. "
            "With the part no longer anchored, the nozzle pushes it around, and filament "
            "is extruded into thin air, forming a tangled mess of plastic strands."
        ),
        "cause_count": 0,
        "causes": [
            {
                "cause": "First layer did not adhere properly",
                "solution": (
                    "A weak first layer is the most common cause of prints lifting off the bed. "
                    "Clean the build plate thoroughly, level the bed, adjust the first layer "
                    "height so the filament is pressed firmly (not too close, not too far), "
                    "and use an appropriate bed temperature for your material (PLA ~60C, "
                    "PETG ~80C, ABS ~100C). Apply a thin layer of hairspray, glue stick, or "
                    "use a textured PEI sheet for better grip."
                ),
            },
            {
                "cause": "Insufficient bed adhesion for the model geometry",
                "solution": (
                    "Parts with small footprints or sharp corners are prone to lifting. "
                    "Add a brim or raft to increase the contact area, slow down the first "
                    "layer print speed (15-25 mm/s), and ensure the enclosure or room "
                    "temperature is stable to prevent uneven cooling and warping that "
                    "rips the part off the bed."
                ),
            },
            {
                "cause": "Print cooling fan blowing too hard on first layers",
                "solution": (
                    "Strong part cooling on the first layers can cause rapid contraction "
                    "and bed separation. Turn the fan off or to 0-30% for the first 2-3 "
                    "layers, especially with ABS/ASA, then ramp it up once the part is "
                    "anchored."
                ),
            },
        ],
        "source": "All3DP / Simplify3D Spaghetti troubleshooting",
    },
    {
        "kb_id": "KB102",
        "problem": "Spaghetti (Nozzle Clog or Under-Extrusion)",
        "is_fdm": True,
        "img_classes": ["Spaghetti"],
        "intro": (
            "A partial or full nozzle clog can cause the filament flow to be inconsistent. "
            "After a period of little or no extrusion, pressure builds up and the filament "
            "suddenly bursts out in an uncontrolled blob or stringy mess, producing "
            "spaghetti-like strands."
        ),
        "cause_count": 0,
        "causes": [
            {
                "cause": "Nozzle partially clogged with burnt filament or debris",
                "solution": (
                    "Perform a cold pull: heat the nozzle to just above the filament's "
                    "glass transition, push filament through, then cool and yank it out to "
                    "remove debris. For stubborn clogs, soak the nozzle in acetone (for PLA) "
                    "or replace the nozzle. Clean the nozzle with a brass brush while hot."
                ),
            },
            {
                "cause": "Filament diameter inconsistency or dust",
                "solution": (
                    "Dust and contaminants on the filament get dragged into the hotend and "
                    "can cause clogs. Use a filament dust filter (foam or cloth) at the "
                    "extruder inlet, store filament in dry sealed bags with desiccant, and "
                    "avoid using filament that has absorbed moisture (which causes popping "
                    "and inconsistent extrusion)."
                ),
            },
            {
                "cause": "Hotend temperature too low for the material",
                "solution": (
                    "If the nozzle is not hot enough, the filament does not melt fully and "
                    "extrusion becomes erratic. Verify the hotend temperature matches the "
                    "material's recommended range and calibrate the actual nozzle "
                    "temperature (thermistors can read low)."
                ),
            },
        ],
        "source": "All3DP / Simplify3D Spaghetti troubleshooting",
    },
    {
        "kb_id": "KB103",
        "problem": "Spaghetti (Filament Tangle or Jam)",
        "is_fdm": True,
        "img_classes": ["Spaghetti"],
        "intro": (
            "When filament tangles on the spool or jams in the extruder path, the printer "
            "may stop feeding filament intermittently. Once the tangle releases, a burst "
            "of filament is extruded into the air, creating spaghetti. "
            "This is distinct from Stringing, which is caused by retraction settings."
        ),
        "cause_count": 0,
        "causes": [
            {
                "cause": "Filament tangled on the spool",
                "solution": (
                    "Never let the filament end go slack on the spool — always secure it "
                    "in the spool holes after use. Check the spool for overlaps and rewind "
                    "neatly. Ensure the spool rotates freely on its holder without binding."
                ),
            },
            {
                "cause": "Extruder gear grinding / stripping the filament",
                "solution": (
                    "If the extruder tension is too high or the nozzle is partially clogged, "
                    "the drive gear can chew through the filament, creating a flat spot that "
                    "slips. Reduce extruder tension slightly, clear any clog, and check that "
                    "the PTFE tube in Bowden setups is fully seated and not kinked."
                ),
            },
            {
                "cause": "Filament path obstruction (kinked PTFE tube)",
                "solution": (
                    "Inspect the full filament path from spool to nozzle. Replace any kinked "
                    "or deformed PTFE tubing, ensure the tube is cut straight at both ends, "
                    "and that there are no sharp bends restricting filament feed."
                ),
            },
        ],
        "source": "All3DP / Simplify3D Spaghetti troubleshooting",
    },
]


def merge_entries(entries, merge_groups):
    """按 MERGE_GROUPS 合并同义条目，合并 causes（去重）。"""
    id_to_entry = {e["kb_id"]: e for e in entries}
    merged_ids = set()
    result = []

    for group in merge_groups:
        primary_id = group[0]
        if primary_id in merged_ids:
            continue
        primary = id_to_entry[primary_id]
        combined_causes = list(primary.get("causes", []))
        seen_cause_text = {c["cause"].strip().lower() for c in combined_causes}

        for other_id in group[1:]:
            other = id_to_entry.get(other_id)
            if not other:
                continue
            merged_ids.add(other_id)
            for c in other.get("causes", []):
                key = c["cause"].strip().lower()
                if key and key not in seen_cause_text:
                    seen_cause_text.add(key)
                    combined_causes.append(c)

        # 更新主条目
        primary["causes"] = combined_causes
        primary["cause_count"] = len(combined_causes)
        primary["merged_from"] = group  # 记录合并来源
        primary["intro"] = primary.get("intro", "") + (
            f"\n\n[Merged from: {', '.join(group)}]" if len(group) > 1 else ""
        )
        result.append(primary)

    # 加入未被合并的条目
    for e in entries:
        if e["kb_id"] not in merged_ids and not any(
            e["kb_id"] in g for g in merge_groups
        ):
            result.append(e)
        elif e["kb_id"] in [g[0] for g in merge_groups] and e not in result:
            pass  # already added above

    # 确保主条目都在 result 里（上面循环可能漏）
    primary_ids = {g[0] for g in merge_groups}
    existing = {e["kb_id"] for e in result}
    for e in entries:
        if e["kb_id"] in primary_ids and e["kb_id"] not in existing:
            result.append(e)

    return result


def main():
    with open(KB_V1_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"原始条目数: {len(data)}")

    # 1. 剔除非 FDM
    data = [e for e in data if e["kb_id"] not in NON_FDM_IDS]
    print(f"剔除非 FDM ({NON_FDM_IDS}) 后: {len(data)}")

    # 2. 同义合并
    merged = merge_entries(data, MERGE_GROUPS)
    merged.sort(key=lambda e: e["kb_id"])
    print(f"同义合并后: {len(merged)} (合并了 {len(data) - len(merged)} 条)")

    # 3. 补充 Spaghetti
    for entry in SPAGHETTI_ENTRIES:
        entry["cause_count"] = len(entry["causes"])
    merged.extend(SPAGHETTI_ENTRIES)
    print(f"补充 Spaghetti {len(SPAGHETTI_ENTRIES)} 条后: {len(merged)}")

    # 4. 统计
    total_causes = sum(e.get("cause_count", 0) for e in merged)
    fdm_count = sum(1 for e in merged if e.get("is_fdm"))
    print(f"最终: {len(merged)} 条问题, {total_causes} 条根因, FDM={fdm_count}")

    with open(KB_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"已写入 {KB_V2_PATH}")


if __name__ == "__main__":
    main()
