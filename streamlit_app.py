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
import time
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
GROQ_API_KEY = _secret_or_env("GROQ_API_KEY")
GROQ_MODEL = _secret_or_env("GROQ_MODEL", "llama-3.3-70b-versatile")


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


_YEAR_RANGE_RE = re.compile(r"(20\d{2})\s*[-–]\s*(20\d{2})")


def sources_matching_academic_year(question: str, chunks: list) -> set:
    """Identify which source files explicitly name the academic year range
    in the question (e.g. '2026-2027').

    Several notices (đăng ký học phần for different semesters) share near-
    identical wording and only differ by year — a weak embedding model
    reranks them almost interchangeably. We don't rely on any single chunk
    repeating the year phrase (chunking may split it away from the table);
    instead we just tag which *documents* mention that year range anywhere,
    then let the caller prefer already-retrieved chunks from those documents.
    """
    match = _YEAR_RANGE_RE.search(question)
    if not match:
        return set()
    year_range = f"{match.group(1)}-{match.group(2)}"
    sources = set()
    for chunk in chunks:
        text = re.sub(r"\s+", "", chunk.page_content.replace("–", "-"))
        if year_range in text and "nămhọc" in text.lower():
            sources.add(chunk.metadata.get("source"))
    return sources


_CONTACT_INTENT_RE = re.compile(r"địa điểm|ở đâu|liên hệ|hotline|số điện thoại", re.IGNORECASE)
_CONTACT_INFO_RE = re.compile(r"hotline|trụ sở|đường dây nóng|địa điểm|địa chỉ", re.IGNORECASE)
_DOT_BATCH_RE = re.compile(r"đợt\s*(\d)\s*/\s*(20\d{2})", re.IGNORECASE)


def find_contact_chunks(question: str, candidates: list, chunks: list) -> list:
    """When the question asks where/how to contact, pin the chunk with the
    actual address/hotline for the source already surfaced by retrieval.

    Contact info is usually a short closing paragraph at the end of a notice
    ('Mọi thông tin liên hệ: ... Hotline: ...'). It shares little semantic
    overlap with a location/contact question compared to the notice's main
    body text, so reranking alone tends to prefer the wrong chunk from the
    same (correct) document.

    Almost every VHU notice ends with this same boilerplate, so multiple
    unrelated candidate sources can match — if the question names a specific
    "đợt N/YYYY" batch, only pin the source whose own text also names that
    exact batch, to avoid mixing in another notice's contact block.
    """
    if not _CONTACT_INTENT_RE.search(question):
        return []
    candidate_sources = {d.metadata.get("source") for d in candidates}

    batch_match = _DOT_BATCH_RE.search(question)
    if batch_match:
        num, year = batch_match.group(1), batch_match.group(2)
        batch_sources = set()
        for c in chunks:
            src = c.metadata.get("source")
            if src not in candidate_sources:
                continue
            text = re.sub(r"\s+", " ", c.page_content.lower())
            if re.search(rf"đợt\s*{num}\b.{{0,15}}{year}|{year}.{{0,15}}đợt\s*{num}\b", text):
                batch_sources.add(src)
        if batch_sources:
            candidate_sources = batch_sources

    hits = []
    for c in chunks:
        if c.metadata.get("source") in candidate_sources and _CONTACT_INFO_RE.search(c.page_content):
            hits.append(c)
    return hits


_TABLE_ROW_RE = re.compile(r"^\s*\d{1,2}\.\s")
_TABLE_HEADER_RE = re.compile(r"đợt\s*1", re.IGNORECASE)


def pin_table_header_chunks(docs: list, chunks: list) -> list:
    """When a selected chunk looks like a mid-table numbered row (e.g. an
    'Đợt 1 / Đợt 2 / Đợt 3' schedule table whose header only appears once
    per page), also pin that source's chunk carrying the column header —
    without it, values in the row can't be mapped to the right 'đợt'.
    """
    by_source: dict = {}
    for c in chunks:
        by_source.setdefault(c.metadata.get("source"), []).append(c)

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in docs}
    extra = []
    for d in docs:
        content = d.page_content
        if not _TABLE_ROW_RE.match(content) or _TABLE_HEADER_RE.search(content[:200]):
            continue
        source = d.metadata.get("source")
        siblings = sorted(by_source.get(source, []), key=lambda c: c.metadata.get("chunk_id", 0))
        for c in siblings:
            if _TABLE_HEADER_RE.search(c.page_content):
                key = (c.metadata.get("source"), c.metadata.get("chunk_id"))
                if key not in seen:
                    extra.append(c)
                    seen.add(key)
                break
    return extra


_LIST_TABLE_HEADER_RE = re.compile(r"STT\s+HỌ\s*TÊN|danh sách giảng viên", re.IGNORECASE)
_DOC_END_MARKER_RE = re.compile(r"TUQ\.|PHÓ TRƯỞNG|HIỆU TRƯỞNG|GIÁM ĐỐC ĐIỀU HÀNH|Nơi nhận:", re.IGNORECASE)


def pin_list_continuation_chunks(docs: list, chunks: list) -> list:
    """When a selected chunk opens a numbered list/table (e.g. a lecturer
    roster with 'STT | HỌ TÊN | EMAIL...' columns) but doesn't yet contain
    the document's closing signature block, the list runs past the chunk's
    character limit into the next chunk(s) of the same source — pin those
    too so the full roster (and its count) reaches the model.
    """
    by_source: dict = {}
    for c in chunks:
        by_source.setdefault(c.metadata.get("source"), []).append(c)
    for src in by_source:
        by_source[src].sort(key=lambda c: c.metadata.get("chunk_id", 0))

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in docs}
    extra = []
    for d in docs:
        if not _LIST_TABLE_HEADER_RE.search(d.page_content):
            continue
        if _DOC_END_MARKER_RE.search(d.page_content):
            continue
        source = d.metadata.get("source")
        cur_id = d.metadata.get("chunk_id")
        for c in by_source.get(source, []):
            cid = c.metadata.get("chunk_id")
            if cid is None or cid <= cur_id:
                continue
            key = (source, cid)
            if key not in seen:
                extra.append(c)
                seen.add(key)
            if _DOC_END_MARKER_RE.search(c.page_content):
                break
    return extra


_VN_WORD_RE = re.compile(
    r"[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+"
)
_STOPWORDS = {
    "và", "là", "của", "vào", "khi", "nào", "cho", "các", "với", "đợt",
    "năm", "tháng", "tuần", "sinh", "viên", "trong", "về", "đến", "được",
}


def pin_relevant_table_rows(question: str, docs: list, chunks: list) -> list:
    """For a source already selected whose chunk looks like a numbered row of
    an 'Đợt 1/Đợt 2/...' schedule table, also pull in sibling row-chunks from
    that same source with strong keyword overlap with the question.

    The row that actually answers a specific 'đợt N' question may sit in a
    different row-chunk than the one the reranker happened to pick — this
    only searches within a source already confirmed relevant by retrieval,
    so the blast radius stays limited to that one document.
    """
    by_source: dict = {}
    for c in chunks:
        by_source.setdefault(c.metadata.get("source"), []).append(c)

    q_words = set(_VN_WORD_RE.findall(question.lower())) - _STOPWORDS

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in docs}
    extra = []
    for d in docs:
        if not _TABLE_ROW_RE.match(d.page_content):
            continue
        source = d.metadata.get("source")
        siblings = by_source.get(source, [])
        if not any(_TABLE_HEADER_RE.search(c.page_content) for c in siblings):
            continue
        for c in siblings:
            key = (c.metadata.get("source"), c.metadata.get("chunk_id"))
            if key in seen or not _TABLE_ROW_RE.match(c.page_content):
                continue
            c_words = set(_VN_WORD_RE.findall(c.page_content.lower()))
            if len(q_words & c_words) >= 2:
                extra.append(c)
                seen.add(key)
    return extra


def _call_gemini(system_text: str, user_text: str) -> str:
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
    return "".join(parts).strip()


def _call_groq(system_text: str, user_text: str) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # Groq's own free-tier rate limit can trip under back-to-back requests
    # (e.g. several students asking in quick succession). Retry with backoff
    # before giving up, since the window is short-lived (per-minute).
    last_exc = None
    for attempt, wait in enumerate((0, 5, 15)):
        if wait:
            time.sleep(wait)
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
        if resp.status_code == 429:
            last_exc = httpx.HTTPStatusError(
                f"Groq 429 (attempt {attempt + 1})", request=resp.request, response=resp
            )
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise last_exc


def ask_llm(question: str, retriever, embeddings, chunks: list) -> tuple[str, list[str], str]:
    pinned = find_by_doc_number(question, chunks)
    candidates = retriever.invoke(question)

    # If the question names a specific academic year (e.g. "2026-2027"),
    # prefer already-retrieved chunks from documents naming that same year
    # over the reranker's opinion — near-duplicate notices from different
    # semesters otherwise get reranked almost interchangeably.
    year_sources = sources_matching_academic_year(question, chunks)
    if year_sources:
        pinned += [d for d in candidates if d.metadata.get("source") in year_sources][:2]

    contact_chunks = find_contact_chunks(question, candidates, chunks)
    pinned += contact_chunks

    reranked = rerank_documents(question, candidates, embeddings, top_k=5) if candidates else []

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in pinned}
    docs = pinned + [
        d for d in reranked if (d.metadata.get("source"), d.metadata.get("chunk_id")) not in seen
    ]
    # A confident contact-info pin already answers the question directly —
    # keep the context small so an unrelated notice (same "tốt nghiệp" topic,
    # different đợt/year) doesn't crowd it out for weaker fallback models.
    docs = docs[: 3 if contact_chunks else 5]
    extra_rows = pin_relevant_table_rows(question, docs, chunks)
    extra_headers = pin_table_header_chunks(docs + extra_rows, chunks)
    extra_continuation = pin_list_continuation_chunks(docs, chunks)
    # Put surgical fix-ups first so they survive format_context's char budget
    # even when the original top-5 already fills most of it.
    docs = extra_rows + extra_headers + extra_continuation + docs
    context = format_context(docs, question)
    messages = build_prompt().format_messages(context=context, question=question)
    system_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") == "system")
    user_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") != "system")

    sources = sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})

    # Gemini is the primary provider (proven accuracy on this corpus). If its
    # free-tier quota is exhausted (HTTP 429), fall back to Groq automatically
    # rather than showing the user an error.
    try:
        answer = _call_gemini(system_text, user_text)
        return answer, sources, "gemini"
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 429 or not GROQ_API_KEY:
            raise
    answer = _call_groq(system_text, user_text)
    return answer, sources, "groq"


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
                answer, sources, provider = ask_llm(question, retriever, embeddings, all_chunks)
            except Exception as exc:
                answer, sources, provider = f"Xin lỗi, có lỗi khi tạo câu trả lời: {exc}", [], None
        st.markdown(answer)
        if sources:
            st.caption("Nguồn: " + ", ".join(sources))
        if provider == "groq":
            st.caption("⚡ Trả lời bởi Groq (Gemini tạm hết quota)")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
