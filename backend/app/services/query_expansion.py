"""Local query expansion for accent-insensitive retrieval."""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, Iterable

from rag_config import VISOLEX_ENABLED


_ABBREVIATION_MAP: tuple[tuple[str, str], ...] = (
    ("nghien cuu khoa hoc", "nghien cuu khoa hoc"),
    ("cong nghe thong tin", "cong nghe thong tin"),
    ("giang vien", "giang vien"),
    ("sinh vien", "sinh vien"),
    ("hoc phan", "hoc phan"),
    ("tin chi", "tin chi"),
    ("tot nghiep", "tot nghiep"),
    ("chung chi", "chung chi"),
    ("ho so", "ho so"),
    ("thong tin", "thong tin"),
    ("nckh", "nghien cuu khoa hoc"),
    ("cntt", "cong nghe thong tin"),
    ("gv", "giang vien"),
    ("sv", "sinh vien"),
    ("hp", "hoc phan"),
    ("tc", "tin chi"),
    ("tn", "tot nghiep"),
    ("cc", "chung chi"),
    ("hs", "ho so"),
    ("tt", "thong tin"),
)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_diacritics(text: str) -> str:
    folded = unicodedata.normalize("NFD", text or "")
    return "".join(char for char in folded if not unicodedata.combining(char)).replace("\u0111", "d").replace("\u0110", "D")


_REQUEST_PREFIX_PATTERN = re.compile(
    r"^(?:(?:theo|hay|vui long|cho (?:minh|toi)|dua tren)\b[^:?,]{0,80}[:;,]\s*)+",
    flags=re.IGNORECASE,
)


def _strip_request_prefix(text: str) -> str:
    """Remove conversational request framing while preserving the factual query."""
    stripped = _REQUEST_PREFIX_PATTERN.sub("", text or "").strip()
    return stripped or (text or "").strip()

def _token_boundary_pattern(source: str) -> re.Pattern[str]:
    escaped = re.escape(source)
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _apply_replacements(text: str) -> tuple[str, list[str]]:
    result = text
    applied: list[str] = []
    for source, target in _ABBREVIATION_MAP:
        pattern = _token_boundary_pattern(source)
        if pattern.search(result):
            result = pattern.sub(target, result)
            if target != source and source not in applied:
                applied.append(source)
    return _collapse_whitespace(result), applied


def _unique_variants(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    variants: list[str] = []
    for value in values:
        candidate = _collapse_whitespace(value)
        if not candidate:
            continue
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    return variants


def normalize_query_for_retrieval(query: str) -> Dict[str, Any]:
    """Build a local retrieval-friendly query without calling external services."""
    result: Dict[str, Any] = {
        "query": query,
        "normalized_query": query,
        "folded_query": query,
        "variants": [query] if query and query.strip() else [],
        "used": False,
        "changed": False,
        "status": "disabled",
        "seconds": 0.0,
        "expansion_applied": [],
    }
    if not VISOLEX_ENABLED or not query.strip():
        return result

    started = time.perf_counter()
    canonical = _collapse_whitespace(query)
    folded = _collapse_whitespace(_strip_diacritics(canonical).casefold())
    retrieval_folded = _strip_request_prefix(folded)
    expanded, applied = _apply_replacements(retrieval_folded)
    best = expanded if applied else retrieval_folded
    variants = _unique_variants([canonical, best, folded, retrieval_folded, expanded])
    result.update(
        normalized_query=best,
        folded_query=folded,
        variants=variants,
        used=best != canonical,
        changed=best != canonical,
        status="ok:expanded" if applied else ("ok:folded" if folded != canonical else "ok"),
        expansion_applied=applied,
        seconds=round(time.perf_counter() - started, 4),
    )
    return result
