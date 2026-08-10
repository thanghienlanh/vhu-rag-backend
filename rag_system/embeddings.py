"""
embeddings.py
Local HuggingFace embeddings setup.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings

from rag_config import EMBEDDING_MODEL

_cached_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings(force_reload: bool = False) -> HuggingFaceEmbeddings:
    """
    Returns local sentence-transformers embeddings (singleton).
    Model will be downloaded on first use (stored in ~/.cache).
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    print(f"Loading embedding model: {model_name} (device={device})...")

    class _QueryAwareEmbeddings(HuggingFaceEmbeddings):
        """BGE models need a retrieval prefix on queries (not documents)."""

        def embed_query(self, text: str) -> list[float]:
            if "bge" in (self.model_name or "").lower():
                text = f"Represent this sentence for searching relevant passages: {text}"
            return super().embed_query(text)

    embeddings = _QueryAwareEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    _ = embeddings.embed_query("sinh viên đăng ký học phần")
    _cached_embeddings = embeddings
    print("Embeddings model loaded successfully.")
    return embeddings
