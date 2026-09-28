"""知识库切分：problem 级 chunk + cause 级 chunk。

每个 problem 生成 1 个 problem chunk（含 intro），每个 cause 生成 1 个 cause chunk。
切分是 RAG 检索的最小粒度单位。
"""
import json
from config import KB_V2_PATH, CHUNKS_PATH


def make_chunk(chunk_id, chunk_type, kb_id, problem, text, extra_meta=None):
    chunk = {
        "chunk_id": chunk_id,
        "type": chunk_type,           # "problem" or "cause"
        "kb_id": kb_id,
        "problem": problem,
        "text": text,
    }
    if extra_meta:
        chunk.update(extra_meta)
    return chunk


def chunk_kb(entries):
    chunks = []
    for e in entries:
        kb_id = e["kb_id"]
        problem = e["problem"]
        intro = e.get("intro", "").strip()

        # problem 级 chunk：问题名 + intro
        problem_text = problem
        if intro:
            problem_text = f"{problem}\n{intro}"
        chunks.append(make_chunk(
            chunk_id=f"{kb_id}_p",
            chunk_type="problem",
            kb_id=kb_id,
            problem=problem,
            text=problem_text,
        ))

        # cause 级 chunk：cause + solution
        for idx, c in enumerate(e.get("causes", [])):
            cause_text = c.get("cause", "").strip()
            solution = c.get("solution", "").strip()
            text = cause_text
            if solution:
                text = f"{cause_text}\n{solution}"
            chunks.append(make_chunk(
                chunk_id=f"{kb_id}_c{idx}",
                chunk_type="cause",
                kb_id=kb_id,
                problem=problem,
                text=text,
                extra_meta={"cause": cause_text},
            ))

    return chunks


def main():
    with open(KB_V2_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    chunks = chunk_kb(entries)
    n_problem = sum(1 for c in chunks if c["type"] == "problem")
    n_cause = sum(1 for c in chunks if c["type"] == "cause")

    print(f"总 chunk 数: {len(chunks)} (problem={n_problem}, cause={n_cause})")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已写入 {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
