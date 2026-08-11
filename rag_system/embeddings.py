"""
embeddings.py
Quantized BGE-M3 embeddings via fastembed (ONNX int8), sized to fit small
deployment targets (~570MB vs ~2.3GB for the full-precision model).

Wraps fastembed.TextEmbedding directly (instead of langchain_community's
FastEmbedEmbeddings) so we can tune ONNX Runtime session memory: disabling
the CPU memory arena and pinning to a single thread trims the extra
allocator/thread-pool overhead on top of the model weights themselves —
every MB matters when the host RAM ceiling sits right at the model size.
"""

import os
from typing import List

from langchain_core.embeddings import Embeddings

_CUSTOM_MODEL_NAME = "Xenova/bge-m3-quantized"
_cached_embeddings: "FastEmbedONNXEmbeddings | None" = None


def _register_quantized_bge_m3() -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    already_registered = any(
        m["model"] == _CUSTOM_MODEL_NAME for m in TextEmbedding.list_supported_models()
    )
    if already_registered:
        return

    TextEmbedding.add_custom_model(
        model=_CUSTOM_MODEL_NAME,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf="Xenova/bge-m3"),
        dim=1024,
        model_file="onnx/model_quantized.onnx",
    )


class FastEmbedONNXEmbeddings(Embeddings):
    """Minimal LangChain-compatible wrapper giving direct access to ONNX
    Runtime session tuning (thread count, memory arena) that the stock
    langchain_community.FastEmbedEmbeddings does not expose.
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
    Returns quantized BGE-M3 embeddings (singleton).
    Model will be downloaded on first use (stored in ~/.cache/fastembed).
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    model_name = os.getenv("EMBEDDING_MODEL", _CUSTOM_MODEL_NAME)
    print(f"Loading embedding model: {model_name} (fastembed, ONNX int8, 1 thread, no mem arena)...")

    if model_name == _CUSTOM_MODEL_NAME:
        _register_quantized_bge_m3()

    embeddings = FastEmbedONNXEmbeddings(model_name=model_name)
    _ = embeddings.embed_query("sinh viên đăng ký học phần")
    _cached_embeddings = embeddings
    print("Embeddings model loaded successfully.")
    return embeddings
