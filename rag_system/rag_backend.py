"""
rag_backend.py
Clean backend interface for using this RAG system as a module from other applications (e.g. Streamlit UI).

Usage from external project:
    import sys
    sys.path.insert(0, r"D:\\NCKH\\rag_system")
    from rag_backend import query, init_backend

    init_backend(
        data_dir=r"D:\\NCKH\\pdfs",
        model="qwen2.5:7b",
        use_hybrid=True
    )
    result = query("Câu hỏi của bạn ở đây?")
    print(result["answer"])
    print(result["sources"])
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure we can import local modules
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Lazy imports - only when needed
_embeddings = None
_vectorstore = None
_chunks_cache = None
_chain = None
_current_config = {}

from rag_config import (
    PAPERS_DIR as DEFAULT_PAPERS_DIR,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    NUM_CTX,
    MAX_OUTPUT_TOKENS,
    USE_HYBRID_SEARCH,
    INITIAL_RETRIEVE_K,
    FINAL_TOP_K,
    USE_RERANKER,
    USE_QUERY_REWRITING,
    NO_INFO_ANSWER,
    USE_RELEVANCE_GUARD,
    MIN_RELEVANCE_SCORE,
    MIN_RELEVANT_CHUNKS,
    NEIGHBOR_CHUNK_WINDOW,
    MAX_CONTEXT_CHUNKS,
)
from embeddings import get_embeddings
from vectorstore import (load_faiss_index, get_retriever, create_faiss_index, index_needs_rebuild, compute_corpus_fingerprint, save_index_meta)
from loader import load_pdfs, load_documents_from_paths
from chunker import chunk_documents
from hybrid_retriever import get_hybrid_retriever
from rag_chain import build_rag_chain, format_context
from reranker import (
    rerank_documents,
    light_keyword_boost_reorder,
    filter_relevant_chunks,
    expand_adjacent_chunks,
    prefer_primary_source,
)
from rag_config import PROJECT_ROOT, FAISS_FULL_PATH


def init_backend(
    data_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    model: Optional[str] = None,
    use_hybrid: Optional[bool] = None,
    embedding_model: Optional[str] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Initialize (or re-initialize) the RAG backend.
    Call this once when your app starts or when user changes settings.

    Returns a dict with status info.
    """
    global _embeddings, _vectorstore, _chunks_cache, _chain, _current_config

    data_dir = data_dir or DEFAULT_PAPERS_DIR
    index_dir = index_dir or str(FAISS_FULL_PATH)
    model = model or OLLAMA_MODEL
    use_hybrid = use_hybrid if use_hybrid is not None else USE_HYBRID_SEARCH
    embedding_model = embedding_model or EMBEDDING_MODEL

    _current_config = {
        "data_dir": str(data_dir),
        "index_dir": str(index_dir),
        "model": model,
        "use_hybrid": use_hybrid,
        "embedding_model": embedding_model,
    }

    print(f"[rag_backend] Initializing with data_dir={data_dir}, model={model}, hybrid={use_hybrid}")

    # 1. Embeddings (shared)
    _embeddings = get_embeddings()

    index_path = Path(index_dir)
    index_file = index_path / "index.faiss"
    corpus_fingerprint = compute_corpus_fingerprint(data_dir)
    needs_rebuild = force_rebuild or index_needs_rebuild(
        index_path,
        embedding_model=embedding_model,
        corpus_fingerprint=corpus_fingerprint,
    )

    # 2. Load or build vectorstore
    if needs_rebuild:
        print("[rag_backend] Building index (this may take time on first run)...")
        raw_docs = load_pdfs(data_dir)
        chunks = chunk_documents(raw_docs, _embeddings)
        _chunks_cache = chunks

        if index_path.exists():
            import shutil
            shutil.rmtree(index_path, ignore_errors=True)
        index_path.mkdir(parents=True, exist_ok=True)

        _vectorstore = create_faiss_index(chunks, _embeddings, index_path, corpus_fingerprint=corpus_fingerprint)

        # Save chunks cache for future fast loads
        import json
        to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
        with open(index_path / "chunks_cache.json", "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False)
        print("  Saved chunks cache for future runs.")
    else:
        print("[rag_backend] Loading existing FAISS index...")
        _vectorstore = load_faiss_index(_embeddings, index_path)
        # Load chunks for BM25 from cache if available (avoids re-reading all PDFs every time)
        chunks_cache_path = index_path / "chunks_cache.json"
        if chunks_cache_path.exists():
            import json
            with open(chunks_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            from langchain_core.documents import Document
            _chunks_cache = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in cached]
            print(f"  Loaded {len(_chunks_cache)} cached chunks for hybrid.")
        else:
            print("  No chunks cache, computing once...")
            raw_docs = load_pdfs(data_dir)
            _chunks_cache = chunk_documents(raw_docs, _embeddings)
            # persist
            import json
            to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in _chunks_cache]
            with open(chunks_cache_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False)
            print("  Saved chunks cache.")

    # 3. Build retriever + chain
    retriever = get_retriever(
        _vectorstore,
        k=INITIAL_RETRIEVE_K,
        use_hybrid=use_hybrid,
        chunks_for_bm25=_chunks_cache,
    )

    _chain = build_rag_chain(
        retriever,
        embeddings=_embeddings,
        final_k=FINAL_TOP_K,
        all_chunks=_chunks_cache,
    )

    print("[rag_backend] Backend ready.")
    return {
        "status": "ready",
        "config": _current_config,
        "num_chunks": len(_chunks_cache) if _chunks_cache else 0,
    }


def query(
    question: str,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
    model: Optional[str] = None,
    history: Optional[List[dict]] = None,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for asking a question using the RAG backend.

    Args:
        question: User question
        use_hybrid: Override hybrid search (default from init)
        filter_source: e.g. "140.MY25" to restrict to files containing this
        model: Override LLM model (currently set at init time)

    Returns:
        {
            "answer": str,
            "sources": list[dict] with 'source', 'page',
            "metadata": {...}
        }
    """
    global _chain, _current_config, _vectorstore, _chunks_cache, _embeddings

    if _chain is None:
        # Auto-init with defaults if not initialized
        print("[rag_backend] Auto-initializing backend...")
        init_backend()

    # Query Rewriting (strongly recommended) - conversation aware
    retrieval_question = question
    if USE_QUERY_REWRITING and not fast_mode:
        try:
            from langchain_ollama import OllamaLLM
            rewriter = OllamaLLM(
                model=OLLAMA_MODEL, 
                timeout=OLLAMA_TIMEOUT,
                options={"num_predict": 128, "num_ctx": NUM_CTX}
            )

            history_text = ""
            if history and len(history) > 0:
                history_lines = []
                for msg in history[-6:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prefix = "User" if role == "user" else "Assistant"
                    history_lines.append(f"{prefix}: {content}")
                history_text = "\n".join(history_lines) + "\n\n"

            rewrite_prompt = f"""You are an expert at rewriting user questions for document retrieval systems over Vietnamese official university documents (PDFs in Vietnamese).

Given the conversation history (if any) and the latest user question, rewrite the latest question into a clear, standalone query that:
- Is self-contained
- Expands abbreviations when helpful (NCKH → nghiên cứu khoa học)
- Is optimized for BOTH semantic search and keyword/BM25 search over Vietnamese academic documents.
- Focus on distinctive phrases that differentiate similar official notices (e.g. "đề tài nghiên cứu khoa học", "thời hạn đăng ký đề tài", "Khoa Công nghệ Thông tin").
- CRITICAL: Keep the rewritten query in the SAME LANGUAGE as the original question. If original is Vietnamese, output in Vietnamese. NEVER translate Vietnamese questions into English.

{history_text}Original: {question}

Rewritten:"""
            rewritten = rewriter.invoke(rewrite_prompt).strip()
            if rewritten and len(rewritten) > 5:
                # Language safeguard
                has_vn_chars = any(c in 'ăâêôơưđĐ' for c in question.lower()) or any(ord(c) > 127 for c in question)
                mostly_en = sum(c.isascii() and c.isalpha() for c in rewritten) > len(rewritten) * 0.6
                if has_vn_chars and mostly_en and ('registration' in rewritten.lower() or 'deadline' in rewritten.lower()):
                    print("[Query Rewriting] Unwanted EN translation detected, using original.")
                else:
                    print(f"[Query Rewriting] {question[:60]}... → {rewritten[:60]}...")
                    retrieval_question = rewritten
        except Exception as e:
            print(f"[Query Rewriting] Skipped: {e}")

    # Build filter if requested
    metadata_filter = {}
    if filter_source:
        metadata_filter["source_contains"] = filter_source

    # If overrides are provided, we may need to rebuild retriever/chain
    need_rebuild = False
    effective_hybrid = use_hybrid if use_hybrid is not None else _current_config.get("use_hybrid", USE_HYBRID_SEARCH)

    if metadata_filter or (use_hybrid is not None and use_hybrid != _current_config.get("use_hybrid")):
        # Re-create retriever with overrides (lightweight)
        retriever = get_retriever(
            _vectorstore,
            k=INITIAL_RETRIEVE_K,
            use_hybrid=effective_hybrid,
            chunks_for_bm25=_chunks_cache,
            metadata_filter=metadata_filter or None,
        )
        chain = build_rag_chain(
            retriever,
            embeddings=_embeddings,
            final_k=FINAL_TOP_K,
            all_chunks=_chunks_cache,
        )
    else:
        chain = _chain

    # Run retrieval using rewritten query (best for recall)
    # But feed the *original* question to the prompt so the LLM sees a natural question
    try:
        tmp_retr = get_retriever(
            _vectorstore,
            k=INITIAL_RETRIEVE_K,
            use_hybrid=effective_hybrid,
            chunks_for_bm25=_chunks_cache,
            metadata_filter=metadata_filter or None,
        )
        # Always retrieve with original question here (hybrid + reranker are now strong enough; avoids bad rewrites)
        raw_docs = tmp_retr.invoke(question)
        boosted_docs = light_keyword_boost_reorder(retrieval_question, raw_docs, top_k=len(raw_docs))
        if USE_RERANKER and _embeddings is not None and len(boosted_docs) > FINAL_TOP_K:
            final_docs = rerank_documents(retrieval_question, boosted_docs, _embeddings, top_k=FINAL_TOP_K)
        else:
            final_docs = boosted_docs[:FINAL_TOP_K]
        # Merge boosted/reranked chunks first; raw top only adds coverage.
        seen = set()
        merged = []
        for d in (final_docs + raw_docs[:5]):
            key = (d.metadata.get('source'), d.metadata.get('chunk_id'), d.page_content[:60])
            if key not in seen:
                seen.add(key)
                merged.append(d)
        raw_docs = merged[:FINAL_TOP_K]
        if USE_RELEVANCE_GUARD and _embeddings is not None and raw_docs:
            from reranker import filter_relevant_chunks as _frc
            raw_docs = _frc(
                retrieval_question,
                raw_docs,
                _embeddings,
                min_score=MIN_RELEVANCE_SCORE,
                min_chunks=MIN_RELEVANT_CHUNKS,
                preserve_order=True,
            )
        raw_docs = expand_adjacent_chunks(
            raw_docs,
            _chunks_cache,
            window=NEIGHBOR_CHUNK_WINDOW,
            max_chunks=MAX_CONTEXT_CHUNKS,
        )
        raw_docs = prefer_primary_source(question, raw_docs)
        ctx = format_context(raw_docs, question)
        from rag_chain import build_prompt as _bp
        from langchain_ollama import OllamaLLM as _OllamaLLM
        from langchain_core.output_parsers import StrOutputParser
        _local_llm = _OllamaLLM(
            model=OLLAMA_MODEL, 
            timeout=OLLAMA_TIMEOUT,
            options={"num_predict": MAX_OUTPUT_TOKENS, "num_ctx": NUM_CTX}
        )
        ans_chain = _bp() | _local_llm | StrOutputParser()
        answer = ans_chain.invoke({"context": ctx, "question": question})
    except Exception as _e:
        # Fallback to old chain behavior
        print('[rag_backend] direct answer path failed, fallback:', _e)
        answer = chain.invoke(retrieval_question)

    # Note: Relevance Guard is applied in the active rag_service path.
    # If you want it here too, you can add filter_relevant_chunks on source_docs below.

    # Get sources for transparency (re-retrieve using the same query used for generation)
    try:
        retriever_for_sources = get_retriever(
            _vectorstore,
            k=FINAL_TOP_K,
            use_hybrid=effective_hybrid,
            chunks_for_bm25=_chunks_cache,
            metadata_filter=metadata_filter or None,
        )
        source_docs = retriever_for_sources.invoke(retrieval_question)
        source_docs = light_keyword_boost_reorder(retrieval_question, source_docs, top_k=len(source_docs))
        source_docs = source_docs[:FINAL_TOP_K]

        # Apply Relevance Guard on source docs
        if USE_RELEVANCE_GUARD and _embeddings is not None and source_docs:
            from reranker import filter_relevant_chunks
            source_docs = filter_relevant_chunks(
                retrieval_question,
                source_docs,
                _embeddings,
                min_score=MIN_RELEVANCE_SCORE,
                min_chunks=MIN_RELEVANT_CHUNKS,
                preserve_order=True,
            )
        source_docs = expand_adjacent_chunks(
            source_docs,
            _chunks_cache,
            window=NEIGHBOR_CHUNK_WINDOW,
            max_chunks=MAX_CONTEXT_CHUNKS,
        )
        source_docs = prefer_primary_source(question, source_docs)
    except Exception:
        source_docs = []

    sources = []
    seen = set()
    for doc in source_docs:
        key = f"{doc.metadata.get('source', 'unknown')}|{doc.metadata.get('page', '?')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id"),
            })

    # Limit to 3 sources max for cleaner UI
    sources = sources[:3]

    # Further prefer sources from the top document to reduce noise from unrelated files
    if sources:
        top_file = sources[0]["source"]
        sources = [s for s in sources if s["source"] == top_file][:3] or sources[:3]

    cleaned_answer = answer
    refusals = [NO_INFO_ANSWER, "Tôi không biết dựa trên tài liệu.", "Tôi không biết dự trên tài liệu.", "I don't know based on the document."]
    is_refusal = False
    for refusal in refusals:
        if refusal in answer:
            before = answer.split(refusal)[0].strip()
            if before and len(before) > 5:
                cleaned_answer = before
            else:
                cleaned_answer = NO_INFO_ANSWER
            is_refusal = True
            break
    return {
        "answer": cleaned_answer,
        "sources": [] if is_refusal else sources,
        "metadata": {
            "model": _current_config.get("model", OLLAMA_MODEL),
            "hybrid": effective_hybrid,
            "filter": metadata_filter or None,
        }
    }


def get_status() -> Dict[str, Any]:
    """Return current backend status."""
    return {
        "initialized": _chain is not None,
        "config": _current_config,
        "chunks_loaded": len(_chunks_cache) if _chunks_cache else 0,
    }


def add_or_update_files(file_paths: list[str]):
    """Smart incremental update: only process new or updated files.

    Much faster than full rebuild when syncing a few files from Google Drive.
    """
    global _vectorstore, _chunks_cache, _embeddings, _current_config

    if not file_paths:
        return

    if _vectorstore is None or not _chunks_cache:
        print("[Warning] No existing index. Falling back to full initialization.")
        init_backend()
        return

    from pathlib import Path as PathlibPath
    import json

    sources_to_remove = {PathlibPath(p).name for p in file_paths}

    # Remove old chunks for these sources
    before_count = len(_chunks_cache)
    _chunks_cache = [c for c in _chunks_cache if c.metadata.get("source") not in sources_to_remove]
    removed_count = before_count - len(_chunks_cache)

    # Load changed files through the same multi-format parser as a full rebuild.
    existing_paths = [path for path in file_paths if os.path.exists(path)]
    new_raw_docs = load_documents_from_paths(existing_paths) if existing_paths else []

    if not new_raw_docs:
        print("No new content found in provided files.")
        return

    new_chunks = chunk_documents(new_raw_docs, _embeddings)
    _chunks_cache.extend(new_chunks)

    # Update the index
    if removed_count > 0:
        # Updates present → safest to recreate from current chunk list
        print(f"Recreating index: removed {removed_count} old chunks, added {len(new_chunks)}...")
        _vectorstore = FAISS.from_documents(_chunks_cache, _embeddings)
    else:
        # Pure new files → fast append
        print(f"Adding {len(new_chunks)} new chunks to existing index...")
        _vectorstore.add_documents(new_chunks)

    # Save updated index and chunks cache
    index_path = Path(_current_config.get("index_dir", str(FAISS_FULL_PATH)))
    _vectorstore.save_local(str(index_path))
    save_index_meta(
        index_path,
        corpus_fingerprint=compute_corpus_fingerprint(_current_config.get("data_dir", DEFAULT_PAPERS_DIR)),
    )

    chunks_cache_path = index_path / "chunks_cache.json"
    import json
    to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in _chunks_cache]
    with open(chunks_cache_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False)

    print(f"Smart update complete. Total chunks: {len(_chunks_cache)}")


def remove_documents(file_names: list[str]):
    """Remove documents by their source filenames from the index and cache.
    Used for hot delete without full rebuild.
    """
    global _vectorstore, _chunks_cache, _embeddings, _current_config

    if not file_names or not _chunks_cache:
        return

    sources_to_remove = set(file_names)
    before_count = len(_chunks_cache)
    _chunks_cache = [c for c in _chunks_cache if c.metadata.get("source") not in sources_to_remove]
    removed_count = before_count - len(_chunks_cache)

    if removed_count > 0:
        if _chunks_cache:
            print(f"Recreating index after removing {removed_count} chunks...")
            _vectorstore = FAISS.from_documents(_chunks_cache, _embeddings)
        else:
            print("Index is now empty after removal.")

        # Save
        index_path = Path(_current_config.get("index_dir", str(FAISS_FULL_PATH)))
        if _vectorstore:
            _vectorstore.save_local(str(index_path))
            save_index_meta(
                index_path,
                corpus_fingerprint=compute_corpus_fingerprint(_current_config.get("data_dir", DEFAULT_PAPERS_DIR)),
            )

        chunks_cache_path = index_path / "chunks_cache.json"
        import json
        to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in _chunks_cache]
        with open(chunks_cache_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False)

        print(f"Remove complete. Total chunks: {len(_chunks_cache)}")


# Convenience for direct testing
if __name__ == "__main__":
    init_backend()
    result = query("Tóm tắt nội dung chính của các văn bản?")
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  - {s['source']} (p.{s['page']})")
