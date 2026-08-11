"""
Configuration for the RAG system.
"""

import os
from pathlib import Path

# === Data & Index Paths ===
# Default: use the local papers/ folder (self-contained project)
PAPERS_DIR: str = str(Path(__file__).parent / "papers")
FAISS_INDEX_DIR: str = "faiss_index"

# === Models ===
# Quantized BGE-M3 (fastembed/ONNX int8, ~570MB) — fits small deployment
# targets. Thay thế: bkai-foundation-models/vietnamese-bi-encoder
_DEFAULT_EMBEDDING = "Xenova/bge-m3-quantized"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
# 3B model nhanh hơn trên CPU; override bằng OLLAMA_MODEL trong .env
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# === Retrieval Settings ===
INITIAL_RETRIEVE_K: int = int(os.getenv("INITIAL_RETRIEVE_K", "20"))          # Tăng recall cho câu hỏi dài
RERANK_MAX_EMBED_DOCS: int = int(os.getenv("RERANK_MAX_EMBED_DOCS", "14"))
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "7"))                  # Nhiều chunk hơn cho bảng / quy định nhiều mục

# Include nearby chunks from the same source after retrieval. This protects
# answers that span adjacent semantic chunks without hard-coding any document.
NEIGHBOR_CHUNK_WINDOW: int = int(os.getenv("NEIGHBOR_CHUNK_WINDOW", "2"))

# === Semantic Chunking ===
SEMANTIC_BREAKPOINT_TYPE: str = "percentile"   # Options: "percentile", "standard_deviation", "interquartile"
SEMANTIC_BREAKPOINT_AMOUNT: float = 90         # Higher = fewer chunks (tune 80-95)

# === Reranking ===
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() in ("1", "true", "yes")

# === Relevance Guard (filters low-relevance chunks after retrieval/rerank) ===
USE_RELEVANCE_GUARD: bool = os.getenv("USE_RELEVANCE_GUARD", "true").lower() in ("1", "true", "yes")
MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.12"))          # Nới cho corpus nhỏ (~22–36 chunk)
MIN_RELEVANT_CHUNKS: int = int(os.getenv("MIN_RELEVANT_CHUNKS", "1"))               # 1 chunk mạnh đủ cho RAG

# === Hybrid Search (BM25 + Semantic) ===
USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
HYBRID_WEIGHTS: list = [float(value.strip()) for value in os.getenv("HYBRID_WEIGHTS", "0.35,0.65").split(",")]   # [semantic_weight, bm25_weight] -- favor BM25/keyword for precise terms like 'đăng ký đề tài NCKH'

# === Metadata Filtering ===
# Example: {"source": "140.MY25.TB.pdf"} or {"source_contains": "MY25"}
DEFAULT_METADATA_FILTER: dict = {}

# === Refusal message (pure RAG: no answer outside retrieved context) ===
NO_INFO_ANSWER: str = "Không tìm thấy thông tin trong tài liệu."

# === Query Rewriting ===
USE_QUERY_REWRITING: bool = True

# Bypasses answer-specific extractors; legacy remains available.
PURE_RAG: bool = os.getenv("PURE_RAG", "false").lower() in ("1", "true", "yes")
PURE_RAG_GENERATION_TIMEOUT: int = int(os.getenv("PURE_RAG_GENERATION_TIMEOUT", "180"))
PURE_RAG_REQUEST_TIMEOUT: int = int(os.getenv("PURE_RAG_REQUEST_TIMEOUT", "240"))

# Local query expansion / accent folding. The compatibility env var is kept for
# benchmark toggles, but the default path no longer depends on an external service.
VISOLEX_ENABLED: bool = os.getenv("VISOLEX_ENABLED", "true").lower() in ("1", "true", "yes")
VISOLEX_URL: str = os.getenv("VISOLEX_URL", "http://127.0.0.1:8011")
VISOLEX_TIMEOUT_SECONDS: float = float(os.getenv("VISOLEX_TIMEOUT_SECONDS", "2.0"))

# === Speed Optimizations (edit these for faster answers) ===
# RECOMMENDED: Set USE_FAST_MODE = True for real-world use
# This is the single biggest speed win: disables rewriting + reranking, uses tiny context
# Accuracy-first for evaluation and production Q&A over official documents.
USE_FAST_MODE: bool = False

MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
NUM_CTX: int = int(os.getenv("NUM_CTX", "4096"))
PURE_RAG_TEMPERATURE: float = float(os.getenv("PURE_RAG_TEMPERATURE", "0"))
PURE_RAG_MAX_OUTPUT_TOKENS: int = int(os.getenv("PURE_RAG_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))
PURE_RAG_NUM_CTX: int = int(os.getenv("PURE_RAG_NUM_CTX", str(NUM_CTX)))
MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "10"))
# Keep the assembled prompt bounded before it reaches the model. Character
# budgeting is conservative because this project has no model-specific tokenizer.
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

if USE_FAST_MODE:
    USE_QUERY_REWRITING = False
    USE_RERANKER = False
    INITIAL_RETRIEVE_K = 10
    FINAL_TOP_K = 4
    MAX_CONTEXT_CHUNKS = 6
    MAX_OUTPUT_TOKENS = 256
    NUM_CTX = 2048
    MAX_CONTEXT_CHARS = min(MAX_CONTEXT_CHARS, 6000)

# === Project root ===
PROJECT_ROOT = Path(__file__).parent.resolve()
FAISS_FULL_PATH = PROJECT_ROOT / FAISS_INDEX_DIR
PAPERS_FULL_PATH = Path(PAPERS_DIR)

# Make sure index dir exists
FAISS_FULL_PATH.mkdir(parents=True, exist_ok=True)
