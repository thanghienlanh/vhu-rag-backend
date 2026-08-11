"""
embeddings.py
Lightweight multilingual embeddings via fastembed (ONNX, ~220MB), sized to
fit free-tier hosting (512MB RAM ceilings). BGE-M3 — even int8-quantized
(~570MB) — does not fit; this model trades some retrieval quality for
guaranteed headroom under low-RAM hosts.
"""

import os
from typing import List

from langchain_core.embeddings import Embeddings

_DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_cached_embeddings: "FastEmbedONNXEmbeddings | None" = None


class FastEmbedONNXEmbeddings(Embeddings):
    """Minimal LangChain-compatible wrapper around fastembed.TextEmbedding,
    with ONNX Runtime session tuning (thread count, memory arena) exposed —
    the stock langchain_community.FastEmbedEmbeddings does not expose these.
    """

    def __init__(self, model_name: str, threads: int = 1, enable_cpu_mem_arena: bool = False):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=model_name,
            threads=threads,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(self.model.query_embed(text)).tolist()


def get_embeddings(force_reload: bool = False) -> FastEmbedONNXEmbeddings:
    """
    Returns multilingual embeddings (singleton).
    Model will be downloaded on first use (stored in ~/.cache/fastembed).
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    model_name = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL_NAME)
    print(f"Loading embedding model: {model_name} (fastembed, ONNX, 1 thread, no mem arena)...")

    embeddings = FastEmbedONNXEmbeddings(model_name=model_name)
    _ = embeddings.embed_query("sinh viên đăng ký học phần")
    _cached_embeddings = embeddings
    print("Embeddings model loaded successfully.")
    return embeddings
