"""
chunker.py
Structure-aware chunking for Vietnamese official PDFs (sections, tables, schedules).
Falls back to semantic chunking only when CHUNK_STRATEGY=semantic.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from rag_config import SEMANTIC_BREAKPOINT_AMOUNT, SEMANTIC_BREAKPOINT_TYPE

CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "structure").strip().lower()
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "120"))
TABLE_MAX_CHARS = int(os.getenv("TABLE_MAX_CHARS", "2000"))

SECTION_START_RE = re.compile(
    r"(?m)(?:^|\n)\s*(\d{1,2})\.\s+(?=[A-ZÀ-ỸĐ\"THÔNG])"
)
SUB_ITEM_RE = re.compile(r"(?m)(?:^|\n)\s*[•\-]\s+")

TABLE_MARKERS = (
    "stt họ tên",
    "stt đợt",
    "email/điện thoại",
    "lĩnh vực hướng dẫn",
    "kế hoạch đăng ký học phần",
    "mã ngành",
    "học phần bổ sung",
)


def get_semantic_chunker(embeddings: HuggingFaceEmbeddings) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
    )


def _is_table_text(text: str) -> bool:
    lowered = (text or "").lower()
    if any(marker in lowered for marker in TABLE_MARKERS):
        return True
    if lowered.count("đợt") >= 2 and "mở" in lowered and "đóng" in lowered:
        return True
    if re.search(r"stt\s", lowered) and re.search(r"\d+\s+[A-ZÀ-ỸĐ]", text or ""):
        return True
    return False


STT_ROW_RE = re.compile(r"(?m)^\s*(\d{1,2})\s+[A-ZÀ-ỸĐ]")


def _first_stt(text: str) -> Optional[int]:
    match = STT_ROW_RE.search(text or "")
    return int(match.group(1)) if match else None


def _last_stt(text: str) -> Optional[int]:
    matches = STT_ROW_RE.findall(text or "")
    return int(matches[-1]) if matches else None


def _is_table_continuation(prev: Document, nxt: Document) -> bool:
    if prev.metadata.get("source") != nxt.metadata.get("source"):
        return False
    prev_text = prev.page_content or ""
    nxt_text = nxt.page_content or ""
    prev_lower = prev_text.lower()
    nxt_lower = nxt_text.lower()
    if not (_is_table_text(prev_lower) or _is_table_text(nxt_lower)):
        return False
    prev_page = int(prev.metadata.get("page", -1))
    next_page = int(nxt.metadata.get("page", -2))
    if next_page != prev_page + 1:
        return False

    last_stt = _last_stt(prev_text)
    first_stt = _first_stt(nxt_text)
    if last_stt is not None and first_stt is not None and first_stt == last_stt + 1:
        return True

    if _is_table_text(prev_lower) and STT_ROW_RE.search(nxt_text):
        return True
    return False


def _attach_preamble_to_parts(preamble: str, parts: List[str], max_chars: int) -> List[str]:
    """Keep letterhead (Số TB, THÔNG BÁO, tiêu đề) on the first chunk."""
    preamble = (preamble or "").strip()
    if not preamble:
        return parts
    if not parts:
        if len(preamble) <= max_chars:
            return [preamble]
        return _split_by_length(preamble, max_chars)

    merged_first = f"{preamble}\n\n{parts[0]}".strip()
    if len(merged_first) <= max_chars:
        return [merged_first, *parts[1:]]

    preamble_parts = (
        [preamble]
        if len(preamble) <= max_chars
        else _split_by_length(preamble, max_chars)
    )
    return [*preamble_parts, *parts]


def _split_by_sections(text: str, max_chars: int) -> List[str]:
    matches = list(SECTION_START_RE.finditer(text))
    if len(matches) <= 1:
        return _split_by_length(text, max_chars)

    preamble = text[: matches[0].start()].strip()
    parts: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        part = text[start:end].strip()
        if not part:
            continue
        if len(part) <= max_chars:
            parts.append(part)
        else:
            parts.extend(_split_by_length(part, max_chars))

    if not parts:
        return _split_by_length(text, max_chars) if len(text) > max_chars else [text]
    return _attach_preamble_to_parts(preamble, parts, max_chars)


def _chunk_limit(text: str) -> int:
    return TABLE_MAX_CHARS if _is_table_text(text) else MAX_CHUNK_CHARS


def _split_table_by_stt_rows(text: str, max_chars: int) -> List[str]:
    rows = list(STT_ROW_RE.finditer(text or ""))
    if len(rows) <= 1:
        return _split_by_length(text, max_chars)

    header_end = rows[0].start()
    header = text[:header_end].strip()
    parts: List[str] = []
    current = header

    for idx, match in enumerate(rows):
        row_start = match.start()
        row_end = rows[idx + 1].start() if idx + 1 < len(rows) else len(text)
        row_text = text[row_start:row_end].strip()
        candidate = f"{current}\n{row_text}".strip() if current else row_text
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = f"{header}\n{row_text}".strip() if header else row_text
        if len(current) > max_chars:
            parts.extend(_split_by_length(current, max_chars))
            current = ""

    if current:
        parts.append(current)
    return parts or [text]


def _split_oversized_text(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    if SECTION_START_RE.search(text):
        parts = _split_by_sections(text, max_chars)
    elif _is_table_text(text) and STT_ROW_RE.search(text):
        parts = _split_table_by_stt_rows(text, max_chars)
    else:
        parts = _split_by_length(text, max_chars)

    if len(parts) == 1 and len(parts[0]) > max_chars:
        return _split_by_length(text, max_chars)
    return parts


def _split_by_length(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            sentences = re.split(r"(?<=[.;!?])\s+", para)
            current = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        parts.append(current)
                    current = sentence
    if current:
        parts.append(current)
    return parts


def _chunk_single_page(doc: Document) -> List[Document]:
    text = (doc.page_content or "").strip()
    if not text:
        return []

    limit = _chunk_limit(text)
    if len(text) <= limit:
        return [Document(page_content=text, metadata=dict(doc.metadata))]

    parts = _split_oversized_text(text, limit)

    return [
        Document(page_content=part, metadata=dict(doc.metadata))
        for part in parts
        if part.strip()
    ]


TRAILING_FRAGMENT_CHARS = int(os.getenv("TRAILING_FRAGMENT_CHARS", "450"))


def _merge_trailing_fragments(chunks: List[Document]) -> List[Document]:
    """Merge short tail chunks (footers, page breaks) into the previous same-source chunk."""
    if len(chunks) < 2:
        return chunks

    merged: List[Document] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.page_content or "") < TRAILING_FRAGMENT_CHARS
            and merged[-1].metadata.get("source") == chunk.metadata.get("source")
        ):
            prev = merged[-1]
            combined = f"{prev.page_content}\n{chunk.page_content}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                page_end = max(
                    int(prev.metadata.get("page_end", prev.metadata.get("page", 0))),
                    int(chunk.metadata.get("page", 0)),
                )
                meta = dict(prev.metadata)
                meta["page_end"] = page_end
                merged[-1] = Document(page_content=combined, metadata=meta)
                continue
        merged.append(chunk)
    return merged


def _merge_small_chunks(chunks: List[Document]) -> List[Document]:
    if not chunks:
        return []

    merged: List[Document] = []
    buffer: Optional[Document] = None

    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue

        buffer_small = len(buffer.page_content or "") < MIN_CHUNK_CHARS
        same_source = buffer.metadata.get("source") == chunk.metadata.get("source")
        combined_len = len(buffer.page_content or "") + len(chunk.page_content or "")

        if buffer_small and same_source and combined_len <= MAX_CHUNK_CHARS:
            buffer = Document(
                page_content=f"{buffer.page_content}\n{chunk.page_content}".strip(),
                metadata=dict(buffer.metadata),
            )
            continue

        merged.append(buffer)
        buffer = chunk

    if buffer is not None:
        merged.append(buffer)
    return merged


def _merge_table_continuations(chunks: List[Document]) -> List[Document]:
    if not chunks:
        return []

    merged: List[Document] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        if idx + 1 < len(chunks) and _is_table_continuation(current, chunks[idx + 1]):
            nxt = chunks[idx + 1]
            combined = f"{current.page_content}\n{nxt.page_content}".strip()
            if len(combined) <= TABLE_MAX_CHARS:
                page_end = max(int(current.metadata.get("page", 0)), int(nxt.metadata.get("page", 0)))
                meta = dict(current.metadata)
                meta["page_end"] = page_end
                current = Document(page_content=combined, metadata=meta)
                idx += 2
            else:
                idx += 1
        else:
            idx += 1
        merged.append(current)
    return merged


def _enforce_chunk_limits(chunks: List[Document]) -> List[Document]:
    finalized: List[Document] = []
    for chunk in chunks:
        text = chunk.page_content or ""
        limit = _chunk_limit(text)
        if len(text) <= limit:
            finalized.append(chunk)
            continue
        for part in _split_oversized_text(text, limit):
            if part.strip():
                finalized.append(Document(page_content=part.strip(), metadata=dict(chunk.metadata)))
    return finalized


def structure_chunk_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        return []

    print(f"Performing structure-aware chunking (max={MAX_CHUNK_CHARS} chars)...")
    raw_chunks: List[Document] = []
    for doc in documents:
        raw_chunks.extend(_chunk_single_page(doc))

    raw_chunks = _merge_small_chunks(raw_chunks)
    raw_chunks = _merge_table_continuations(raw_chunks)
    raw_chunks = _merge_trailing_fragments(raw_chunks)
    raw_chunks = _enforce_chunk_limits(raw_chunks)
    raw_chunks = _merge_small_chunks(raw_chunks)

    for i, chunk in enumerate(raw_chunks):
        chunk.metadata["chunk_id"] = i
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

    print(f"Created {len(raw_chunks)} structure-aware chunks.")
    return raw_chunks


def semantic_chunk_documents(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    if not documents:
        return []

    print("Performing semantic chunking...")
    chunker = get_semantic_chunker(embeddings)
    chunks = chunker.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

    chunks = _merge_small_chunks(chunks)
    print(f"Created {len(chunks)} semantic chunks.")
    return chunks


def chunk_documents(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    if CHUNK_STRATEGY == "semantic":
        return semantic_chunk_documents(documents, embeddings)
    return structure_chunk_documents(documents)