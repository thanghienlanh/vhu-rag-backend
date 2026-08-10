"""
Pre-retrieval query guards: normalize messy user input and refuse off-corpus queries.

Design: regex/normalization catches common variants; retrieval + prompt handle the rest.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional, Sequence

_YEAR_RE = re.compile(r"20\d{2}")

_ACADEMIC_YEAR_SPAN_RE = re.compile(
    r"20\d{2}\s*[-–—/]\s*20\d{2}|năm\s+học\s+20\d{2}|học\s+kỳ\s+.*20\d{2}"
)

_ARITH_OP_RE = re.compile(r"(\d+)\s*[\+\*×÷/＋]\s*(\d+)")
_ARITH_MINUS_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_VN_ARITH_RE = re.compile(
    r"\d+\s*(?:trừ|cộng|nhân|chia|plus|minus|times|divided\s+by)\s*\d+",
    re.IGNORECASE,
)
_CN_ARITH_RE = re.compile(r"\d+\s*[加減乘除]\s*\d+|\d+等于")

_CORPUS_DOMAIN_MARKERS = (
    "đăng ký",
    "học phần",
    "nckh",
    "đề tài",
    "sinh viên",
    "khoa",
    "thông báo",
    "giảng viên",
    "học kỳ",
    "tuyển sinh",
    "văn bản",
    "vhu",
    "văn hiến",
    "tốt nghiệp",
    "xét tốt nghiệp",
    "học phí",
    "ielts",
    "toefl",
    "hsk",
    "chứng chỉ",
    "bằng tốt nghiệp",
    "nhận bằng",
    "cấp bằng",
    "khóa 20",
    "đợt ",
    "tín chỉ",
    "song ngành",
    "combo",
    "portal",
    "cntt",
    "công nghệ thông tin",
    "kế toán",
    "kiểm toán",
    "thạc sĩ",
    "nghiên cứu khoa học",
    "lịch thi",
    "thi tự luận",
    "hồ sơ",
    "tình huống",
)

_TRIVIA_MARKERS = (
    "thủ đô",
    "capital of",
    "thời tiết",
    "weather",
    "bóng đá",
    "world cup",
    "tổng thống",
    "president",
    "bitcoin",
    "giá vàng",
    "chiến tranh",
    "kể chuyện",
    "tell me a joke",
    "bạn là ai",
    "who are you",
    "xin chào",
    "hello",
)

_OFFTOPIC_MARKERS = (
    "không liên quan",
    "hoàn toàn không liên quan",
    "tổng thống",
    "president",
    "thời tiết",
    "bóng đá",
    "world cup",
    "ai là vua",
    "chiến tranh",
    "giá vàng",
    "bitcoin",
)

_MATH_RESULT_MARKERS = ("bằng mấy", "等于多少", "等于几", "equals", "kết quả", "tính", "calculate")

_corpus_max_year_cache: Optional[int] = None


def normalize_query(text: str) -> str:
    """Unify unicode dashes/operators and collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    for src, dst in (
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2212", "-"),
        ("＋", "+"),
        ("×", "*"),
        ("÷", "/"),
    ):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def extract_years(text: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(normalize_query(text))]


def query_in_corpus_domain(query: str) -> bool:
    q = normalize_query(query).lower()
    return any(marker in q for marker in _CORPUS_DOMAIN_MARKERS)


def is_academic_year_span(query: str) -> bool:
    q = normalize_query(query).lower()
    if re.search(r"2026\s*[-–—/]\s*2027", q) or "năm học 2026" in q:
        return True
    return bool(_ACADEMIC_YEAR_SPAN_RE.search(q))


def infer_corpus_max_year(chunk_texts: Optional[Iterable[str]] = None) -> int:
    """Max 20xx year seen in indexed chunks; fallback 2027 for current VHU corpus."""
    global _corpus_max_year_cache
    if chunk_texts is None:
        if _corpus_max_year_cache is not None:
            return _corpus_max_year_cache
        return 2027

    years: list[int] = []
    for text in chunk_texts:
        years.extend(extract_years(text or ""))
    if not years:
        max_year = 2027
    else:
        counts = Counter(years)
        # Ignore single OCR typos like 07/8/2028 in schedule tables.
        plausible = [y for y in years if y <= 2027 or counts[y] >= 2]
        max_year = max(plausible) if plausible else 2027
    _corpus_max_year_cache = max_year
    return max_year


def is_unsupported_future_query(
    query: str,
    *,
    max_corpus_year: Optional[int] = None,
) -> bool:
    """Refuse years clearly outside indexed documents (dynamic bound from corpus)."""
    q = normalize_query(query).lower()
    years = extract_years(q)
    if not years:
        return False

    # Always refuse explicit post-corpus years (e.g. "học kỳ 1 năm 2028").
    if any(y >= 2028 for y in years):
        return True

    if is_academic_year_span(q):
        return False

    ceiling = (max_corpus_year if max_corpus_year is not None else infer_corpus_max_year()) + 1
    if any(y >= ceiling for y in years):
        return True
    if f"năm {ceiling}" in q or str(ceiling) in q:
        return True
    return False


def _is_date_slash_fragment(query: str, match: re.Match[str]) -> bool:
    """True for dd/mm inside dd/mm/yyyy, or obvious day/month (e.g. 30/11)."""
    tail = query[match.end() : match.end() + 5]
    if tail.startswith("/20"):
        return True
    try:
        left_i, right_i = int(match.group(1)), int(match.group(2))
    except ValueError:
        return False
    return 13 <= left_i <= 31 and 1 <= right_i <= 12


def _is_math_operand_pair(left: str, right: str) -> bool:
    """True only for real arithmetic — not đợt 2/2025 or 2026-2027."""
    if len(right) == 4 and right.startswith("20"):
        return False
    if (
        len(left) == 4
        and len(right) == 4
        and left.startswith("20")
        and right.startswith("20")
    ):
        return False
    try:
        left_i, right_i = int(left), int(right)
    except ValueError:
        return False
    return left_i < 1000 and right_i < 1000


def looks_like_arithmetic(query: str) -> bool:
    """Detect math expressions; ignore academic year ranges like 2025-2026."""
    q = normalize_query(query).lower()

    for match in _ARITH_OP_RE.finditer(q):
        if _is_date_slash_fragment(q, match):
            continue
        if _is_math_operand_pair(match.group(1), match.group(2)):
            return True
    if _VN_ARITH_RE.search(q):
        return True
    if _CN_ARITH_RE.search(q):
        return True

    for match in _ARITH_MINUS_RE.finditer(q):
        left, right = match.group(1), match.group(2)
        if (
            len(left) == 4
            and len(right) == 4
            and left.startswith("20")
            and right.startswith("20")
        ):
            continue
        if int(left) < 1000 and int(right) < 1000:
            return True

    if any(marker in q for marker in _MATH_RESULT_MARKERS):
        for match in _ARITH_OP_RE.finditer(q):
            if _is_date_slash_fragment(q, match):
                continue
            if _is_math_operand_pair(match.group(1), match.group(2)):
                return True

    return False


def is_arithmetic_or_trivia_query(query: str) -> bool:
    """Math, chitchat, and general knowledge — not in indexed PDFs."""
    raw = normalize_query(query)
    q = raw.lower()
    if not q:
        return False

    if looks_like_arithmetic(q):
        return True

    if any(marker in q for marker in ("bằng mấy", "等于多少", "等于几")) and re.search(r"\d", q):
        if not query_in_corpus_domain(q):
            return True

    if any(marker in q for marker in _MATH_RESULT_MARKERS) and looks_like_arithmetic(q):
        return True

    if not query_in_corpus_domain(q):
        if any(marker in q for marker in _TRIVIA_MARKERS):
            return True
        if re.search(r"[\u4e00-\u9fff]", raw) and not any(
            x in q for x in ("hsk", "tiếng trung", "phụ lục")
        ):
            return True

    return False


def is_offtopic_query(query: str) -> bool:
    """Clearly unrelated questions that should never be answered from university PDFs."""
    q = normalize_query(query).lower()
    if is_arithmetic_or_trivia_query(query):
        return True
    if any(marker in q for marker in _OFFTOPIC_MARKERS):
        return not query_in_corpus_domain(q)
    return False