"""P0 阶段全局配置。

密钥来源优先级：系统环境变量 > 项目根目录 .env 文件。
.env 已被 .gitignore 排除，不会入库。
"""
import os
from pathlib import Path

# 项目根目录（由本文件位置推导，保证可移植）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_env_file(path: Path) -> None:
    """轻量 .env 加载器（不依赖 python-dotenv）。

    已存在的环境变量优先，不被文件覆盖——避免覆盖 CI / 生产注入值。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_env_file(PROJECT_ROOT / ".env")

# 输入 / 输出文件
KB_V1_PATH = DATA_DIR / "kb_v1.json"            # 原始知识库（75 条）
KB_V2_PATH = DATA_DIR / "kb_v2.json"            # 清洗后知识库（去重 + Spaghetti 补充）
CHUNKS_PATH = DATA_DIR / "chunks.json"          # 切分后的 chunk 列表
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"   # BM25 索引
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin" # FAISS 向量索引
CHROMA_DB_PATH = DATA_DIR / "chroma_db"         # Chroma 持久化目录（元数据过滤服务层）
CHROMA_COLLECTION = "print_chunks"              # Chroma collection 名
EVAL_SET_PATH = DATA_DIR / "eval_set.json"      # 200 条评测集

# 检索参数
# 权重经消融调优实测（eval_set 的 gold 修正后重算，2026-09-28）：
#   TF-IDF+SVD 兜底 → 0.5 : 0.5（hybrid Recall@5 = 78.12%，与 BM25 单路持平：同质融合零增量）
#   DashScope 真 embedding → 最优 0.6 : 0.4（hybrid Recall@5 = 85.00%）
# 当前默认取 DashScope 调优值；切回 TF-IDF 兜底时请改回 0.5 / 0.5。
BM25_WEIGHT = 0.6            # BM25 路权重
VECTOR_WEIGHT = 0.4          # 向量路权重
TOP_K = 5                    # 返回 top-k 结果
RECALL_K = 5                 # Recall@k
VECTOR_BACKEND = "faiss"     # 向量后端："faiss"（默认）或 "chroma"（支持 where 元数据过滤）

# Embedding（DashScope text-embedding）
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024             # text-embedding-v3 输出维度
EMBED_BATCH_SIZE = 10        # ⚠️ 单次上限 10 条，超过报 400 InvalidParameter
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 切分策略：problem 级 + cause 级
# 每个 problem 生成 1 个 problem chunk，每个 cause 生成 1 个 cause chunk
CAUSE_CHUNK_MIN_LEN = 0      # 不过滤短 cause，保留全部根因
