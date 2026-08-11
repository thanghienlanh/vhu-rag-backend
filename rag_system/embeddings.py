"""
embeddings.py
Quantized BGE-M3 embeddings via fastembed (ONNX int8), sized to fit small
deployment targets (~570MB vs ~2.3GB for the full-precision model).
"""

import os

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

_CUSTOM_MODEL_NAME = "Xenova/bge-m3-quantized"
_cached_embeddings: FastEmbedEmbeddings | None = None


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


def get_embeddings(force_reload: bool = False) -> FastEmbedEmbeddings:
    """
    Returns quantized BGE-M3 embeddings (singleton).
    Model will be downloaded on first use (stored in ~/.cache/fastembed).
    """
    global _cached_embeddings
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    model_name = os.getenv("EMBEDDING_MODEL", _CUSTOM_MODEL_NAME)
    print(f"Loading embedding model: {model_name} (fastembed, ONNX int8)...")

    if model_name == _CUSTOM_MODEL_NAME:
        _register_quantized_bge_m3()

    embeddings = FastEmbedEmbeddings(model_name=model_name)
    _ = embeddings.embed_query("sinh viên đăng ký học phần")
    _cached_embeddings = embeddings
    print("Embeddings model loaded successfully.")
    return embeddings
