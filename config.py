from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "rag.db"

APP_NAME = "local_rag_simple"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"

TOP_K = 3
MIN_RELEVANCE_SCORE = 0.50
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
