"""
Extract structured NCKH facts (deadline, group size, eligibility) from indexed PDF chunks.
No hardcoded answers — parse only; return None when evidence is missing.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from retrieval_rules import TYPE_NCKH_CNTT, get_chunk_type, strip_doc_display_prefix


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


_QUERY_CANONICAL_PHRASES = ('thời hạn', 'hạn đăng ký', 'đăng ký', 'đề tài', 'nghiên cứu khoa học', 'nhóm', 'tối đa', 'mấy sinh viên', 'bao nhiêu sinh viên', 'điều kiện', 'năm thứ', 'đủ điều kiện', 'được không', 'có đăng ký')


def _fold_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFD", _normalize(text))
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace("đ", "d")


def _normalize_query(text: str) -> str:
    """Augment a query with canonical Vietnamese phrases without changing document parsing."""
    original = _normalize(text)
    folded = _fold_ascii(text)
    canonical = [phrase for phrase in _QUERY_CANONICAL_PHRASES if _fold_ascii(phrase) in folded]
    aliases = {
        "nckh": "nghiên cứu khoa học", "cntt": "công nghệ thông tin",
        "hp": "học phần", "tn": "tốt nghiệp", "hs": "hồ sơ", "hk": "học kỳ",
    }
    for alias, expanded in aliases.items():
        if re.search(rf"(?<![a-z0-9]){alias}(?![a-z0-9])", folded):
            canonical.append(expanded)
    return " ".join([original, folded, *canonical])


def is_nckh_info_query(query: str) -> bool:
    q = _normalize_query(query)
    if not any(x in q for x in ["nckh", "nghiên cứu khoa học", "đề tài", "đăng ký đề tài"]):
        return False
    return any(
        x in q
        for x in [
            "thời hạn",
            "hạn đăng ký",
            "deadline",
            "đến ngày",
            "nhóm",
            "tối đa",
            "mấy sinh viên",
            "bao nhiêu sinh viên",
            "điều kiện",
            "năm 1",
            "năm thứ",
            "năm 2",
            "đủ điều kiện",
            "được không",
            "có đăng ký",
        ]
    )


def is_nckh_deadline_query(query: str) -> bool:
    q = _normalize_query(query)
    if not any(x in q for x in ["nckh", "nghiên cứu", "đề tài"]):
        return False
    return any(x in q for x in ["thời hạn", "hạn đăng ký", "deadline", "đến ngày"])


def _is_nckh_overview_chunk(doc: Document) -> bool:
    if get_chunk_type(doc) != TYPE_NCKH_CNTT:
        return False
    text = _normalize(doc.page_content or "")
    markers = (
        "đối tượng đăng ký",
        "thời hạn đăng ký",
        "đăng ký đề tài nghiên cứu khoa học",
        "nhóm nghiên cứu",
    )
    return sum(1 for marker in markers if marker in text) >= 2


@lru_cache(maxsize=1)
def load_index_chunks_from_disk() -> Tuple[Document, ...]:
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
        print(f"[NCKH Extract] Failed to load chunks_cache.json: {exc}")
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


def _collect_nckh_text(all_docs: List[Document]) -> Tuple[str, str]:
    selected = [doc for doc in all_docs if _is_nckh_overview_chunk(doc)]
    if not selected:
        return "", ""
    selected.sort(
        key=lambda d: (
            d.metadata.get("source") or "",
            int(d.metadata.get("page") or 0),
            int(d.metadata.get("chunk_id") or 0),
        )
    )
    source = str(selected[0].metadata.get("source") or "")
    return "\n".join(doc.page_content or "" for doc in selected), source


def parse_nckh_facts(text: str) -> Dict[str, str]:
    raw = unicodedata.normalize("NFC", text or "")
    facts: Dict[str, str] = {}

    deadline = re.search(
        r"thời hạn đăng ký\s*:\s*[^\n]*?(\d{1,2}/\d{1,2}/\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if deadline:
        facts["deadline"] = deadline.group(1)
    else:
        fallback_deadline = re.search(
            r"hết ngày\s+(\d{1,2}/\d{1,2}/\d{4})",
            raw,
            flags=re.IGNORECASE,
        )
        if fallback_deadline:
            facts["deadline"] = fallback_deadline.group(1)

    group = re.search(
        r"mỗi nhóm nghiên cứu tối đa\s*(\d{2})\s*sinh viên",
        raw,
        flags=re.IGNORECASE,
    )
    if group:
        facts["max_group"] = group.group(1)

    eligibility = re.search(
        r"năm thứ\s*2\s*hoặc\s*3\s*trở lên",
        raw,
        flags=re.IGNORECASE,
    )
    if eligibility:
        facts["eligibility"] = eligibility.group(0).strip()

    return facts


def format_nckh_answer(query: str, facts: Dict[str, str], source: str) -> Optional[str]:
    if not source:
        return None

    q = _normalize_query(query)
    lines: List[str] = []

    if (
        is_nckh_deadline_query(query)
        or "thời hạn" in q
        or "hạn đăng ký" in q
        or re.search(r"\d{1,2}/\d{1,2}/\d{4}", q)
        or any(x in q for x in ["trước hạn", "sau hạn", "nộp hồ sơ", "chưa nộp"])
    ):
        deadline = facts.get("deadline")
        if not deadline:
            return None
        lines.append(
            f"Theo tài liệu **{source}**, thời hạn đăng ký đề tài NCKH "
            f"là **từ ngày ra thông báo đến hết ngày {deadline}**."
        )
    elif any(x in q for x in ["nhóm", "tối đa", "mấy sinh viên", "bao nhiêu sinh viên"]):
        max_group = facts.get("max_group")
        if not max_group:
            return None
        lines.append(
            f"Theo tài liệu **{source}**, mỗi nhóm nghiên cứu NCKH tối đa **{max_group} sinh viên** "
            "và phải có giảng viên hướng dẫn trong Khoa."
        )
    elif any(x in q for x in ["năm 1", "năm thứ", "điều kiện", "đủ điều kiện", "được không"]):
        eligibility = facts.get("eligibility")
        if not eligibility:
            return None
        if any(x in q for x in ["năm 1", "năm thứ 1", "sv năm 1", "sinh viên năm 1"]):
            lines.append(
                f"**Không.** Theo tài liệu **{source}**, sinh viên năm 1 chưa đủ điều kiện đăng ký NCKH. "
                f"Đối tượng là sinh viên Khoa CNTT các hệ đào tạo **{eligibility}**."
            )
        else:
            lines.append(
                f"Theo tài liệu **{source}**, đối tượng đăng ký đề tài NCKH là sinh viên Khoa CNTT "
                f"các hệ đào tạo **{eligibility}**."
            )
    else:
        parts = []
        if facts.get("deadline"):
            parts.append(f"hạn đăng ký đến {facts['deadline']}")
        if facts.get("max_group"):
            parts.append(f"tối đa {facts['max_group']} sinh viên/nhóm")
        if facts.get("eligibility"):
            parts.append(f"đối tượng {facts['eligibility']}")
        if not parts:
            return None
        lines.append(f"Theo tài liệu **{source}**, quy định NCKH: {'; '.join(parts)}.")

    lines.append(f"\nNguồn: {source}.")
    return "\n".join(lines).strip()


def try_extract_nckh_facts(
    query: str,
    docs: List[Document],
    chunks_cache: Optional[List[Document]] = None,
) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    if not is_nckh_info_query(query):
        return None

    merged = _merge_chunk_sources(docs, chunks_cache)
    text, source = _collect_nckh_text(merged)
    facts = parse_nckh_facts(text)
    if not text or not source or not facts:
        return None

    answer = format_nckh_answer(query, facts, source)
    if not answer:
        return None

    overview_chunks = [
        d for d in merged if _is_nckh_overview_chunk(d) and d.metadata.get("page") is not None
    ]
    by_page: Dict[int, Document] = {}
    for d in overview_chunks:
        by_page.setdefault(int(d.metadata.get("page")), d)
    sources = [
        {"source": source, "page": str(page), "text": strip_doc_display_prefix(by_page[page].page_content)}
        for page in sorted(by_page)[:2]
    ] or [{"source": source, "page": "?"}]
    return answer, sources