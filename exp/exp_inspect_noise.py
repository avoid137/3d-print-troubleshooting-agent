"""实查 wrong/partial 案例的证据构成：噪声来自 top-5 还是 parent-child 扩展？"""
import sys, json
sys.path.insert(0, "src")
from config import EVAL_SET_PATH
from skill_diagnose import DiagnoseSkill

skill = DiagnoseSkill()
es = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
by_id = {it["id"]: it for it in es}

# 取诊断里点名的案例 + 几个边界
inspect = ["eval_001","eval_005","eval_007","eval_013","eval_022",
           "eval_027","eval_035","eval_044","eval_167","eval_180"]
for eid in inspect:
    it = by_id.get(eid)
    if not it: 
        continue
    out = skill.run(it["query"], top_k=5, use_judge=False)
    gold = set(it["gold"])
    print("="*72)
    print(f"{eid} | cat={it['category']} | gold={sorted(gold)}")
    print(f"  Q: {it['query']}")
    for e in out["evidence"]:
        tag = "EXPANDED" if e.get("expanded") else "retrieved"
        mark = "GOLD" if e["kb_id"] in gold else "---"
        print(f"    [{e['chunk_id']}] ({tag}) {mark} kb={e['kb_id']} prob={e['problem'][:30]}")
    print(f"  kb_ids(去重)={out['kb_ids']}")
