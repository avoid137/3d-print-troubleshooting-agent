"""BM25 索引：基于 rank_bm25，对 chunk 文本建立精确术语检索。"""
import json
import pickle
import re
from config import CHUNKS_PATH, BM25_INDEX_PATH


def tokenize(text):
    """简单分词：小写 + 非字母数字切分 + 去停用词。"""
    stop = set("""a an the is are was were be been being to of in on at for with by
    from as into onto upon and or but if then than so that this these those it its
    has have had do does did not no can will would should could may might must your
    you we they them their our ours my mine me i he she his her him it's its' don't
    """.split())
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in stop and len(t) > 1]


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 索引已写入 {BM25_INDEX_PATH} ({len(chunks)} chunks)")
    return bm25


def load_bm25():
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def query_bm25(bm25, query, top_k=5):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, scores[i]) for i in top_idx]


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    build_bm25(chunks)


if __name__ == "__main__":
    main()
