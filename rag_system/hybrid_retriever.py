"""
hybrid_retriever.py
Implements Hybrid Search: BM25 (keyword) + FAISS (semantic) using Reciprocal Rank Fusion (RRF).
No dependency on EnsembleRetriever (which is unavailable in some LangChain versions).

This significantly improves retrieval for technical/academic PDFs by combining
keyword exact matches (great for "đăng ký đề tài NCKH", dates, codes) with semantic.
"""

from typing import List, Optional, Dict
from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from accent_bm25 import AccentInsensitiveBM25

from rag_config import INITIAL_RETRIEVE_K, FINAL_TOP_K, HYBRID_WEIGHTS


def create_bm25_retriever(documents: List[Document], k: int = None) -> BM25Retriever:
    """
    Create a BM25 keyword-based retriever from the list of documents/chunks.
    """
    if k is None:
        k = INITIAL_RETRIEVE_K

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = int(k)
    return bm25_retriever


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Document]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Supports optional per-list weights.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for rlist, w in zip(ranked_lists, weights):
        for rank, doc in enumerate(rlist, 1):
            # Use a stable key: prefer chunk_id + source + page + short hash of content
            key = (
                str(doc.metadata.get("chunk_id", ""))
                + "|" + doc.metadata.get("source", "")
                + "|" + str(doc.metadata.get("page", ""))
                + "|" + doc.page_content[:80]
            )
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] += w * (1.0 / (k + rank))

    # Sort by fused score desc
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def create_hybrid_retriever(
    vectorstore: FAISS,
    bm25_documents: Optional[List[Document]] = None,
    weights: List[float] = None,
    k: int = None,
):
    """
    Hybrid retriever using BM25 + semantic with RRF (no EnsembleRetriever required).
    """
    if k is None:
        k = INITIAL_RETRIEVE_K

    if weights is None:
        weights = HYBRID_WEIGHTS or [0.65, 0.35]

    # How many candidates to fetch from each before fusion
    # Lower for speed (was higher for recall)
    fetch_k = int(max(k * 2.5, 12)) if k < 10 else int(max(k * 1.5, 12))

    # Prepare BM25
    if bm25_documents is None or len(bm25_documents) == 0:
        try:
            bm25_documents = list(getattr(vectorstore, "docstore", None)._dict.values()) if hasattr(vectorstore, "docstore") else []
        except Exception:
            bm25_documents = []

    if not bm25_documents:
        print("[Hybrid] No BM25 docs available, falling back to pure semantic.")
        return vectorstore.as_retriever(search_kwargs={"k": k})

    bm25 = AccentInsensitiveBM25(bm25_documents)

    class _HybridRRFRetriever:
        def __init__(self, vs, bm25_ret, rrf_k, rrf_weights, out_k, fetch_k):
            self.vs = vs
            self.bm25_ret = bm25_ret
            self.rrf_k = rrf_k
            self.rrf_weights = rrf_weights
            self.out_k = out_k
            self.fetch_k = fetch_k

        def invoke(self, query: str):
            # Semantic
            try:
                sem_docs = self.vs.similarity_search(query, k=self.fetch_k)
            except Exception:
                sem_docs = []
            # BM25
            try:
                bm_docs = self.bm25_ret.search(query, self.fetch_k)
            except Exception:
                bm_docs = []

            fused = _reciprocal_rank_fusion([sem_docs, bm_docs], k=self.rrf_k, weights=self.rrf_weights)
            return fused[: self.out_k]

        def get_relevant_documents(self, query: str):
            return self.invoke(query)

    return _HybridRRFRetriever(vectorstore, bm25, 60, weights, k, fetch_k)


def get_hybrid_retriever(
    vectorstore: FAISS,
    chunks_for_bm25: List[Document],
    use_hybrid: bool = True,
    k: int = None,
):
    """
    Factory: hybrid (RRF) or pure semantic.
    """
    if not use_hybrid:
        if k is None:
            k = INITIAL_RETRIEVE_K
        return vectorstore.as_retriever(search_kwargs={"k": k})

    return create_hybrid_retriever(
        vectorstore=vectorstore,
        bm25_documents=chunks_for_bm25,
        k=k,
    )
