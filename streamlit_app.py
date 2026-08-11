"""
streamlit_app.py
Standalone chat UI for the VHU RAG system, built for Streamlit Community
Cloud (1GB RAM free tier). Runs the retrieval pipeline in-process (no
separate FastAPI backend) and calls Gemini directly for generation.
"""

import json
import os
import re
import sys
from pathlib import Path

import httpx
import streamlit as st

HERE = Path(__file__).parent.resolve()
RAG_SYSTEM_DIR = HERE / "rag_system"
if str(RAG_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_SYSTEM_DIR))

from embeddings import get_embeddings  # noqa: E402
from vectorstore import (  # noqa: E402
    compute_corpus_fingerprint,
    create_faiss_index,
    get_retriever,
    index_needs_rebuild,
    load_faiss_index,
)
from loader import load_documents as load_pdfs  # noqa: E402
from chunker import chunk_documents  # noqa: E402
from rag_chain import build_prompt, format_context  # noqa: E402
from rag_config import EMBEDDING_MODEL, FAISS_FULL_PATH, PAPERS_DIR  # noqa: E402
from reranker import rerank_documents  # noqa: E402

st.set_page_config(page_title="VHU Document Assistant", page_icon="📚")

def _secret_or_env(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_API_KEY = _secret_or_env("GEMINI_API_KEY")
GEMINI_MODEL = _secret_or_env("GEMINI_MODEL", "gemini-2.5-flash-lite")


@st.cache_resource(show_spinner="Đang tải dữ liệu và mô hình (chỉ chạy 1 lần)...")
def init_retriever():
    embeddings = get_embeddings()
    index_path = Path(FAISS_FULL_PATH)
    corpus_fingerprint = compute_corpus_fingerprint(PAPERS_DIR)

    if index_needs_rebuild(index_path, EMBEDDING_MODEL, corpus_fingerprint=corpus_fingerprint):
        raw_docs = load_pdfs(PAPERS_DIR)
        chunks = chunk_documents(raw_docs, embeddings)
        vectorstore = create_faiss_index(
            chunks, embeddings, index_path, corpus_fingerprint=corpus_fingerprint
        )
        to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
        with open(index_path / "chunks_cache.json", "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False)
    else:
        vectorstore = load_faiss_index(embeddings, index_path)
        from langchain_core.documents import Document

        with open(index_path / "chunks_cache.json", "r", encoding="utf-8") as f:
            cached = json.load(f)
        chunks = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in cached]

    retriever = get_retriever(vectorstore, k=8, use_hybrid=True, chunks_for_bm25=chunks)
    return retriever, embeddings, chunks


_DOC_NUMBER_RE = re.compile(r"(\d{1,3})\s*/\s*(MYH\d{2}|MY\d{2})", re.IGNORECASE)


def find_by_doc_number(question: str, chunks: list) -> list:
    """Exact-match lookup by official notice number (e.g. '91/MYH26').

    This is independent of the embedding model: a weak/light model can rank
    the right chunk poorly for a specific-number query, so we short-circuit
    with a direct string match against the doc_number metadata whenever the
    question clearly names a document number.
    """
    match = _DOC_NUMBER_RE.search(question)
    if not match:
        return []
    number, year_code = match.group(1), match.group(2).upper()
    hits = []
    seen_sources = set()
    for chunk in chunks:
        doc_number = (chunk.metadata.get("doc_number") or "").upper()
        if not doc_number:
            continue
        if doc_number.startswith(f"{number}/") and year_code in doc_number:
            source = chunk.metadata.get("source")
            if source not in seen_sources:
                seen_sources.add(source)
            hits.append(chunk)
    return hits


def ask_gemini(question: str, retriever, embeddings, chunks: list) -> tuple[str, list[str]]:
    pinned = find_by_doc_number(question, chunks)
    candidates = retriever.invoke(question)
    reranked = rerank_documents(question, candidates, embeddings, top_k=5) if candidates else []

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in pinned}
    docs = pinned + [
        d for d in reranked if (d.metadata.get("source"), d.metadata.get("chunk_id")) not in seen
    ]
    docs = docs[:5]
    context = format_context(docs, question)
    messages = build_prompt().format_messages(context=context, question=question)
    system_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") == "system")
    user_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") != "system")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, params={"key": GEMINI_API_KEY}, json=payload)
        resp.raise_for_status()
        data = resp.json()

    parts = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                parts.append(part["text"])

    sources = sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
    return "".join(parts).strip(), sources


st.title("📚 VHU Document Assistant")
st.caption("Trợ lý hỏi-đáp thông báo học vụ — Trường Đại học Văn Hiến")

if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY. Vào Settings → Secrets trên Streamlit Cloud để thêm.")
    st.stop()

retriever, embeddings, all_chunks = init_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Nguồn: " + ", ".join(msg["sources"]))

if question := st.chat_input("Hỏi về học phần, tuyển sinh, tốt nghiệp..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời..."):
            try:
                answer, sources = ask_gemini(question, retriever, embeddings, all_chunks)
            except Exception as exc:
                answer, sources = f"Xin lỗi, có lỗi khi tạo câu trả lời: {exc}", []
        st.markdown(answer)
        if sources:
            st.caption("Nguồn: " + ", ".join(sources))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
