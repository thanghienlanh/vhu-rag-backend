"""
vectorstore.py
FAISS vector database management.
Supports pure semantic and hybrid (BM25 + semantic) retrievers + basic metadata filtering.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_config import FAISS_FULL_PATH, INITIAL_RETRIEVE_K, EMBEDDING_MODEL

INDEX_META_FILE = "index_meta.json"
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".xlsx", ".txt", ".md"}


def compute_corpus_fingerprint(data_dir: str | Path) -> str:
    """Stable manifest hash for supported source files, without reading their content."""
    root = Path(data_dir)
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    files = sorted(
        (item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES),
        key=lambda item: item.as_posix().lower(),
    )
    for item in files:
        stat = item.stat()
        relative = item.relative_to(root).as_posix()
        digest.update(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def save_index_meta(index_path: Path, embedding_model: str = None, corpus_fingerprint: str = None) -> None:
    meta = {
        "embedding_model": embedding_model or EMBEDDING_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": corpus_fingerprint,
    }
    (index_path / INDEX_META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index_meta(index_path: Path) -> Optional[str]:
    meta_path = index_path / INDEX_META_FILE
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data.get("embedding_model")
    except (json.JSONDecodeError, OSError):
        return None


def index_needs_rebuild(index_path: Path, embedding_model: str = None, corpus_fingerprint: str = None) -> bool:
    """True when index is missing or was built with a different embedding model."""
    if not (index_path / "index.faiss").exists():
        return True
    stored = load_index_meta(index_path)
    current = embedding_model or EMBEDDING_MODEL
    if stored is None:
        return True
    if stored != current:
        return True
    if corpus_fingerprint is not None:
        meta_path = index_path / INDEX_META_FILE
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True
        return meta.get("corpus_fingerprint") != corpus_fingerprint
    return False


def create_faiss_index(
    chunks: List[Document],
    embeddings: Embeddings,
    index_path: Path = None,
    corpus_fingerprint: str = None,
) -> FAISS:
    """
    Create FAISS index from semantic chunks and save it.
    """
    if index_path is None:
        index_path = FAISS_FULL_PATH

    print(f"Creating FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_path))
    save_index_meta(index_path, corpus_fingerprint=corpus_fingerprint)
    print(f"FAISS index saved to: {index_path}")
    return vectorstore


def load_faiss_index(
    embeddings: Embeddings,
    index_path: Path = None,
    corpus_fingerprint: str = None,
) -> FAISS:
    """
    Load existing FAISS index from disk.
    """
    if index_path is None:
        index_path = FAISS_FULL_PATH

    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Please run ingestion first."
        )

    print(f"Loading FAISS index from {index_path}...")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded successfully.")
    return vectorstore


def filter_documents_by_metadata(
    docs: List[Document],
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Post-filter documents by metadata.
    Supports:
        {"source": "exact_filename.pdf"}
        {"source_contains": "substring"}
        {"page": 5}
    """
    if not metadata_filter or not docs:
        return docs

    filtered = []
    for doc in docs:
        meta = doc.metadata
        match = True

        for key, value in metadata_filter.items():
            if key == "source_contains":
                if value.lower() not in meta.get("source", "").lower():
                    match = False
                    break
            elif key == "source":
                if meta.get("source") != value:
                    match = False
                    break
            else:
                # Direct match for page or custom fields
                if meta.get(key) != value:
                    match = False
                    break

        if match:
            filtered.append(doc)

    return filtered


def get_retriever(
    vectorstore: FAISS,
    k: int = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = False,
    chunks_for_bm25: Optional[List[Document]] = None,
):
    """
    Return a retriever (pure semantic or hybrid).

    If metadata_filter is provided, we will retrieve extra documents and filter afterwards.
    """
    if k is None:
        k = INITIAL_RETRIEVE_K

    base_k = int(k * 2) if metadata_filter else int(k)   # Fetch more when filtering

    if use_hybrid and chunks_for_bm25:
        from hybrid_retriever import create_hybrid_retriever
        base_retriever = create_hybrid_retriever(
            vectorstore, bm25_documents=chunks_for_bm25, k=base_k
        )
    else:
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": base_k})

    # If no filter needed, return directly
    if not metadata_filter:
        return base_retriever

    # Wrap to apply metadata filter after retrieval
    def filtered_retriever(query: str):
        docs = base_retriever.invoke(query)
        return filter_documents_by_metadata(docs, metadata_filter)[:k]

    # Make it act like a retriever
    class _FilteredRetriever:
        def invoke(self, query: str):
            return filtered_retriever(query)

        def get_relevant_documents(self, query: str):
            return self.invoke(query)

    return _FilteredRetriever()
