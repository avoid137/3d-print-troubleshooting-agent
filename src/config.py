"""P0 阶段全局配置。"""
import os
from pathlib import Path

# 项目根目录（基于 config.py 位置动态计算，换目录不用改）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 输入 / 输出文件
KB_V1_PATH = DATA_DIR / "kb_v1.json"            # 原始知识库（75 条）
KB_V2_PATH = DATA_DIR / "kb_v2.json"            # 清洗后知识库（去重 + Spaghetti 补充）
CHUNKS_PATH = DATA_DIR / "chunks.json"          # 切分后的 chunk 列表
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"   # BM25 索引
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin" # FAISS 向量索引
EVAL_SET_PATH = DATA_DIR / "eval_set.json"      # 200 条评测集

# 检索参数（权重经消融调优，当前为 TF-IDF+SVD 兜底最优；
# 切换 DashScope embedding 后建议重新调优，向量侧权重可上调）
BM25_WEIGHT = 0.5            # BM25 路权重
VECTOR_WEIGHT = 0.5          # 向量路权重
TOP_K = 5                    # 返回 top-k 结果
RECALL_K = 5                 # Recall@k

# Embedding（DashScope text-embedding）
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024             # text-embedding-v3 输出维度
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 切分策略：problem 级 + cause 级
# 每个 problem 生成 1 个 problem chunk，每个 cause 生成 1 个 cause chunk
CAUSE_CHUNK_MIN_LEN = 0      # 不过滤短 cause，保留全部根因
