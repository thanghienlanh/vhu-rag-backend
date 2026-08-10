"""
Extract NCKH lecturer tables dynamically from indexed PDF chunks (no hardcoded lists).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from retrieval_rules import TYPE_NCKH_CNTT, TYPE_NCKH_KHOA, get_chunk_type, strip_doc_display_prefix

STT_ENTRY_RE = re.compile(
    r"(?:^|\n)\s*(\d{1,2})\s+"
    r"((?:[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ\s'.-]{2,}?))\s+"
    r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    re.MULTILINE,
)

NOISE_LINE_RE = re.compile(
    r"^(stt|họ tên|email|lĩnh vực|ghi chú|danh sách giảng viên|\[tài liệu)",
    re.IGNORECASE,
)


def is_nckh_lecturer_query(query: str) -> bool:
    q = (query or "").lower()
    has_gv = "giảng viên" in q or "gvhd" in q
    has_guide = "hướng dẫn" in q or "nhận hướng dẫn" in q
    has_nckh = any(
        x in q
        for x in [
            "nckh",
            "nghiên cứu khoa học",
            "nghiên cứu",
            "đề tài",
            "khoa học sinh viên",
        ]
    )
    if has_gv and (has_guide or has_nckh):
        return True
    if "danh sách" in q and has_gv:
        return True
    if re.search(r"giảng\s*viên.*hướng\s*dẫn", q):
        return True
    return False


def _is_lecturer_chunk(doc: Document) -> bool:
    text = (doc.page_content or "").lower()
    src = (doc.metadata.get("source") or "").lower()
    ctype = get_chunk_type(doc)
    has_email = "@" in text
    stt_matches = len(list(STT_ENTRY_RE.finditer(doc.page_content or "")))
    has_lecturer_marker = any(
        marker in text
        for marker in (
            "danh s?ch gi?ng vi?n",
            "stt h? t?n",
            "email/?i?n tho?i",
            "h??ng d?n",
        )
    )
    if ctype in (TYPE_NCKH_KHOA, TYPE_NCKH_CNTT):
        if has_email and (has_lecturer_marker or stt_matches >= 2):
            return True
    if "kcntt" in src or "09myh26" in src:
        if has_email and stt_matches >= 2:
            return True
    return False


@lru_cache(maxsize=1)
def load_index_chunks_from_disk() -> Tuple[Document, ...]:
    """Load all indexed chunks from faiss_index/chunks_cache.json (no RAG init required)."""
    try:
        from rag_config import FAISS_FULL_PATH

        path = Path(FAISS_FULL_PATH) / "chunks_cache.json"
        if not path.exists():
            return tuple()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}) or {})
            for item in raw
        )
    except Exception as exc:
        print(f"[Extract] Failed to load chunks_cache.json: {exc}")
        return tuple()


def _merge_chunk_sources(
    docs: List[Document],
    chunks_cache: Optional[List[Document]] = None,
) -> List[Document]:
    merged: List[Document] = list(docs or [])
    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id")) for d in merged}

    for source in (chunks_cache or []) + list(load_index_chunks_from_disk()):
        key = (source.metadata.get("source"), source.metadata.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def _collect_lecturer_text(all_docs: List[Document]) -> str:
    selected: List[Document] = []
    for doc in all_docs:
        if _is_lecturer_chunk(doc):
            selected.append(doc)

    if not selected:
        return ""

    selected.sort(
        key=lambda d: (
            d.metadata.get("source") or "",
            int(d.metadata.get("page") or 0),
            int(d.metadata.get("chunk_id") or 0),
        )
    )
    return "\n".join(doc.page_content or "" for doc in selected)


def parse_lecturers_from_text(text: str) -> List[Dict[str, Any]]:
    matches = list(STT_ENTRY_RE.finditer(text or ""))
    if not matches:
        return []

    lecturers: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        stt = int(match.group(1))
        if stt < 1 or stt > 20:
            continue
        name = re.sub(r"\s+", " ", match.group(2)).strip(" .")
        email = match.group(3).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        tail = text[start:end]

        field_lines: List[str] = []
        for raw_line in tail.splitlines():
            line = raw_line.strip()
            if not line or NOISE_LINE_RE.match(line):
                continue
            if STT_ENTRY_RE.match("\n" + line):
                break
            if re.match(r"^\d{1,2}\s+[A-ZÀ-ỸĐ]", line):
                break
            field_lines.append(line)

        fields = re.sub(r"\s+", " ", " ".join(field_lines)).strip()
        lecturers.append({"stt": stt, "name": name, "email": email, "fields": fields})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in sorted(lecturers, key=lambda row: row["stt"]):
        key = (item["stt"], item["email"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_lecturer_answer(lecturers: List[Dict[str, Any]], source: str) -> str:
    lines = [
        f"Theo tài liệu **{source}**, danh sách **{len(lecturers)}** giảng viên nhận hướng dẫn đề tài NCKH:"
    ]
    for item in lecturers:
        block = f"\n**{item['stt']}. {item['name']}** — {item['email']}"
        if item.get("fields"):
            block += f"\n   Lĩnh vực: {item['fields']}"
        lines.append(block)
    return "\n".join(lines).strip()


def try_extract_nckh_lecturers(
    query: str,
    docs: List[Document],
    chunks_cache: Optional[List[Document]] = None,
    min_entries: int = 14,
) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    """Parse lecturer rows from retrieved/cache chunks. No static JSON."""
    if not is_nckh_lecturer_query(query):
        return None

    merged_cache = _merge_chunk_sources(docs, chunks_cache)
    text = _collect_lecturer_text(merged_cache)
    lecturers = parse_lecturers_from_text(text)

    if not text or len(lecturers) < min_entries:
        return None

    used_chunks = [d for d in merged_cache if _is_lecturer_chunk(d)]
    source = ""
    for doc in used_chunks:
        source = str(doc.metadata.get("source") or "")
        if source:
            break

    by_page: Dict[int, Document] = {}
    for d in used_chunks:
        if d.metadata.get("page") is not None:
            by_page.setdefault(int(d.metadata.get("page")), d)

    answer = format_lecturer_answer(lecturers, source)
    sources = [
        {"source": source, "page": str(page), "text": strip_doc_display_prefix(by_page[page].page_content)}
        for page in sorted(by_page)[:3]
    ] or [{"source": source, "page": "?"}]
    return answer, sources
