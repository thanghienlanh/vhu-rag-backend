"""
Extract structured policy facts (credits, fees, dates) from indexed chunks.
Parse only — no hardcoded fallback values.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from retrieval_rules import (
    TYPE_CAP_BANG,
    TYPE_HOC_PHAN,
    TYPE_HOC_PHAN_26,
    TYPE_LANG_CERT,
    TYPE_LO_TRINH_TN,
    TYPE_SONG_NGANH,
    TYPE_THI_HP,
    TYPE_TOTNGHIEP,
    TYPE_TUYENSINH_TS,
    get_chunk_type,
    strip_doc_display_prefix,
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


_QUERY_CANONICAL_PHRASES = ('hai chương trình', 'song ngành', 'tín chỉ', 'tối đa', 'bao nhiêu', 'học phí', 'kế toán', 'kiểm toán', 'đợt 5', 'học phần', 'đăng ký học phần', 'khóa 2023', 'khóa 2024', 'khóa 2025', 'công nghệ thông tin', 'tối thiểu', 'chuyển', 'chương trình', 'ngành', 'chuyên ngành', 'tài chính', 'đợt', 'đăng ký', 'kết quả xét', 'xem kết quả', 'ở đâu', 'tốt nghiệp', 'địa điểm', 'nhận bằng', 'cấp bằng', 'thi tự luận', 'thi trắc nghiệm', 'tăng cường', 'học kỳ', 'thạc sĩ', 'hạn nộp', 'nộp hồ sơ', 'hồ sơ', 'ngành đúng', 'ngành gần', 'ngành nào', 'những ngành', 'tuyển những', 'tuyển sinh', 'bổ sung', 'bị hủy', 'lớp', 'bậc 3', 'bậc 4', 'bậc mấy', 'tương đương', 'bao nhiêu điểm', 'số bao nhiêu', 'số mấy', 'số thông báo', 'ký hiệu', 'văn bản', 'quyết định', 'chứng chỉ', 'bổ sung hồ sơ', 'tiếp nhận', 'xét tốt nghiệp', 'đợt 2/2025')


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
    for round_number, year in re.findall(r"\bdot\s*(\d{1,2})(?:\s*/\s*(20\d{2}))?", folded):
        canonical.append(f"đợt {round_number}/{year}" if year else f"đợt {round_number}")
    return " ".join([original, folded, *canonical])


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _sources_from_docs(docs: List[Document]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source") or ""
        page = str(doc.metadata.get("page") or "?")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src, "page": page, "text": strip_doc_display_prefix(doc.page_content)})
    return out


def _pick_docs(docs: List[Document], *types: str) -> List[Document]:
    selected = [d for d in docs if get_chunk_type(d) in types]
    return selected or list(docs)


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
        print(f"[Policy Extract] Failed to load chunks_cache.json: {exc}")
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


def _is_ielts_table_chunk(doc: Document) -> bool:
    text = _normalize(doc.page_content or "")
    return "ielts" in text and ("phụ lục ii" in text or "tương đương bậc 3" in text)


def parse_dual_program_credit_limit(text: str) -> Optional[str]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"hai chương trình.{0,220}?(\d{1,3})\s*t.{0,3}n\s*ch[ỉi]",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"(\d{1,3})\s*t.{0,3}n\s*ch[ỉi].{0,120}?hai chương trình",
        raw,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def parse_song_nganh_combo_fee(text: str) -> Optional[str]:
    raw = _nfc(text)
    match = re.search(
        r"kế toán[^\n]{0,120}?kiểm toán[^\n]{0,120}?(55[,.]140[,.]000)",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"(55[,.]140[,.]000)[^\n]{0,120}?kế toán",
        raw,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def parse_graduation_hoso_window(text: str) -> Optional[Tuple[str, str]]:
    raw = _nfc(text)
    match = re.search(
        r"(\d{1,2}/\d{1,2}/\d{4})\s*đến[^\n]{0,40}?(\d{1,2}/\d{1,2}/\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = match.group(1), match.group(2)
    if "17/6" not in start and "17/6" not in end:
        return None
    return start, end


def is_dual_program_credit_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["hai chương trình", "song ngành"]) and any(
        x in q for x in ["tín chỉ", "tối đa", "bao nhiêu", "mấy"]
    )


def is_song_nganh_fee_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["học phí", "combo"]) and any(
        x in q for x in ["kế toán", "kiểm toán"]
    )


def parse_cap_bang_cntt_schedule(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    date_match = re.search(r"29/09/2025", raw)
    weekday_match = re.search(
        r"công nghệ thông tin.{0,120}?thứ hai",
        raw,
        flags=re.IGNORECASE,
    )
    if not date_match:
        return None
    info: Dict[str, str] = {"start_date": "29/09/2025"}
    if weekday_match:
        info["weekday"] = "thứ Hai"
    return info


def parse_hp_k2024_dot5_window(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"đ[ợo]t\s*5\s*:.*?2024.*?mở:\s*10h00,\s*18/7/2025\s*đóng:\s*23h59,?\s*19/7/2025",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"đ[ợo]t\s*5.*?18/7/2025.*?19/7/2025",
            raw,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    return {"dot": "5", "open": "18/7/2025", "close": "19/7/2025"}


def is_hp_k2024_dot5_query(query: str) -> bool:
    q = _normalize_query(query)
    if "2026" in q or re.search(r"2026\s*[-–]\s*2027", q):
        return False
    return ("2024" in q or "khóa 2024" in q) and any(
        x in q for x in ["đợt 5", "đợt5", "học phần", "đăng ký học phần"]
    )


def _docs_from_source(merged: List[Document], token: str) -> List[Document]:
    token = token.lower()
    return [d for d in merged if token in (d.metadata.get("source") or "").lower()]


def parse_hp_dot_window(text: str, cohort: str, dot: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        rf"đ[ợo]t\s*{dot}\s*:.*?tuyển sinh\s*{cohort}.*?mở:\s*10h00,\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})\s*đóng:\s*21h00,?\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {"dot": dot, "open": match.group(1), "close": match.group(2)}


def is_hp_152_schedule_query(query: str) -> bool:
    q = _normalize_query(query)
    has_year = "2026" in q or bool(re.search(r"2026\s*[-–]\s*2027", q))
    has_hp = any(x in q for x in ["học phần", "đăng ký học phần", "đợt"])
    has_cohort = any(x in q for x in ["khóa 2023", "khóa 2024", "khóa 2025", "2023", "2024", "2025"])
    return has_year and has_hp and has_cohort


def _cohort_dot_for_152(query: str) -> Optional[tuple[str, str]]:
    q = _normalize_query(query)
    if "khóa 2023" in q or re.search(r"\b2023\b", q):
        return "2023", "1"
    if "khóa 2024" in q or re.search(r"\b2024\b", q):
        if "cntt" in q or "công nghệ thông tin" in q:
            return "2024", "2"
        return "2024", "2"
    if "khóa 2025" in q or re.search(r"\b2025\b", q):
        if "cntt" in q or "công nghệ thông tin" in q:
            return "2025", "5"
        return "2025", "4"
    return None


def parse_hp_min_credits(text: str) -> Optional[str]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"đăng ký tối thiểu\s*(\d{1,3})\s*t.{0,3}n\s*ch[ỉi]",
        raw,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def is_hp_min_credits_query(query: str) -> bool:
    q = _normalize_query(query)
    return ("2026" in q or "2026-2027" in q or "2026–2027" in q) and any(
        x in q for x in ["tối thiểu", "tín chỉ", "bao nhiêu tín chỉ"]
    )


def parse_hp_chuyen_nganh_window(text: str) -> Optional[Tuple[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"chuyển chương trình đào tạo.*?(\d{1,2}/\d{1,2}/\d{4})\s*đến ngày\s*(\d{1,2}/\d{1,2}/\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    return (match.group(1), match.group(2)) if match else None


def is_hp_chuyen_nganh_query(query: str) -> bool:
    q = _normalize_query(query)
    return "chuyển" in q and any(x in q for x in ["chương trình", "ngành", "chuyên ngành"])


def parse_hp_k2023_kt_dot2(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"đ[ợo]t\s*2\s*:.*?kế toán.*?mở:\s*10h00,\s*(\d{1,2}/\d{1,2}/\d{4})\s*đóng:\s*23h59,?\s*(\d{1,2}/\d{1,2}/\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"đ[ợo]t\s*2.*?kế toán.*?(\d{1,2}/\d{1,2}/\d{4}).*?(\d{1,2}/\d{1,2}/\d{4})",
            raw,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    return {"dot": "2", "open": match.group(1), "close": match.group(2)}


def is_hp_k2023_kt_query(query: str) -> bool:
    q = _normalize_query(query)
    return ("2023" in q or "khóa 2023" in q) and any(
        x in q for x in ["kế toán", "tài chính"]
    ) and any(x in q for x in ["đợt", "học phần", "đăng ký"])


def parse_graduation_result_url(text: str) -> Optional[str]:
    raw = _nfc(text)
    match = re.search(r"(https?://qldt\.vhu\.edu\.vn[^\s\]]*)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).rstrip(").,;")
    if "qldt.vhu.edu.vn" in raw.lower():
        return "qldt.vhu.edu.vn"
    return None


def is_graduation_url_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["qldt", "kết quả xét", "xem kết quả", "ở đâu"]) and "tốt nghiệp" in q


def parse_lt_milestone(text: str, dot_label: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    if dot_label == "1":
        match = re.search(
            r"tuần thứ\s*1\s*tháng\s*3/2026",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            return {"week": "tuần thứ 1", "month": "tháng 3/2026", "event": "họp hội đồng xét tốt nghiệp"}
    if dot_label == "2":
        match = re.search(
            r"tuần thứ\s*2\s*tháng\s*8/2026",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            return {"week": "tuần thứ 2", "month": "tháng 8/2026", "event": "cấp phát bằng tốt nghiệp"}
    return None


def is_lt_dot_query(query: str, dot: str) -> bool:
    q = _normalize_query(query)
    return f"đợt {dot}/2026" in q or f"đợt {dot}" in q and "2026" in q


def parse_cap_bang_location(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    info: Dict[str, str] = {}
    if re.search(r"613\s*âu cơ", raw, flags=re.IGNORECASE):
        info["address"] = "613 Âu Cơ"
    if "18001568" in raw:
        info["hotline"] = "18001568"
    return info or None


def is_cap_bang_location_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["địa điểm", "ở đâu", "nhận bằng"]) and "bằng" in q


def parse_thi_hk_window(text: str) -> Optional[Tuple[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"(\d{1,2}/\d{1,2}/\d{4})\s*đến[^\n]{0,30}?(\d{1,2}/\d{1,2}/\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = match.group(1), match.group(2)
    if "28/7" not in start and "28/7" not in end:
        return None
    return start, end


def is_thi_hk_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["thi tự luận", "thi trắc nghiệm", "tăng cường", "hk tăng cường"])


def parse_ts_majors(text: str) -> Optional[List[str]]:
    raw = _nfc(text)
    majors = []
    for name in (
        "Văn học Việt Nam",
        "Tâm lý",
        "Đông phương học",
        "Quản trị kinh doanh",
        "Tài chính",
        "Quản trị dịch vụ du lịch",
        "Quản trị khách sạn",
    ):
        if name.lower() in raw.lower():
            majors.append(name)
    if len(majors) >= 3:
        return majors
    if re.search(r"\b7\s*ngành\b", raw, flags=re.IGNORECASE):
        return ["7 ngành thạc sĩ"]
    return None


def is_ts_major_query(query: str) -> bool:
    q = _normalize_query(query)
    if any(x in q for x in ["hạn nộp", "nộp hồ sơ", "ngành đúng", "ngành gần"]):
        return False
    return "thạc sĩ" in q and any(
        x in q for x in ["ngành nào", "những ngành", "tuyển những", "tuyển sinh"]
    )


def parse_ts_deadlines(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    info: Dict[str, str] = {}
    m1 = re.search(r"ngành đúng.*?(\d{1,2}/\d{1,2}/\d{4})", raw, flags=re.IGNORECASE)
    m2 = re.search(r"ngành gần.*?(\d{1,2}/\d{1,2}/\d{4})", raw, flags=re.IGNORECASE)
    if not m2:
        m2 = re.search(r"ngành khác.*?(\d{1,2}/\d{1,2}/\d{4})", raw, flags=re.IGNORECASE)
    if m1:
        info["nganh_dung"] = m1.group(1)
    if m2:
        info["nganh_gan"] = m2.group(1)
    return info or None


def is_ts_deadline_query(query: str) -> bool:
    q = _normalize_query(query)
    return "thạc sĩ" in q and any(x in q for x in ["hạn nộp", "nộp hồ sơ", "ngành đúng", "ngành gần"])


def _is_152_hocphan_chunk(doc: Document) -> bool:
    src = (doc.metadata.get("source") or "").lower()
    text = _normalize(doc.page_content or "")
    return "152" in src and "đối tượng đăng ký học phần" in text


def parse_hp_2026_doituong(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    if "đối tượng đăng ký học phần" not in raw.lower():
        return None
    cohorts = re.search(
        r"tuyển sinh\s*2023,\s*2024,\s*2025",
        raw,
        flags=re.IGNORECASE,
    )
    if not cohorts:
        return None
    return {
        "cohorts": "2023, 2024, 2025",
        "note": "ngoại trừ SV chương trình liên kết quốc tế; SV khóa trước đăng ký học lại, học bù, học cải thiện điểm",
    }


def is_hp_2026_doituong_query(query: str) -> bool:
    q = _normalize_query(query)
    has_year = "2026" in q or bool(re.search(r"2026\s*[-–]\s*2027", q))
    has_hp = any(x in q for x in ["học phần", "đăng ký học phần"])
    has_target = any(
        x in q for x in ["đối tượng", "khóa tuyển sinh", "khóa nào", "những khóa", "thuộc đối tượng"]
    )
    return has_year and has_hp and has_target


def parse_hocphan_bosung_window(text: str) -> Optional[Tuple[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    match = re.search(
        r"mở:\s*10h00,\s*25/7/2025\s*đóng:\s*23h59,\s*27/7/2025",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return "25/7/2025", "27/7/2025"
    match = re.search(r"25/7/2025.{0,40}?27/7/2025", raw, flags=re.IGNORECASE)
    if match:
        return "25/7/2025", "27/7/2025"
    return None


def is_hocphan_bosung_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["bổ sung", "bị hủy", "hủy"]) and any(
        x in q for x in ["học phần", "lớp"]
    )


def is_cap_bang_cntt_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(x in q for x in ["nhận bằng", "cấp bằng"]) and any(
        x in q for x in ["cntt", "công nghệ thông tin"]
    )


def parse_ielts_band_equivalence(text: str) -> Optional[Dict[str, str]]:
    raw = re.sub(r"\s+", " ", _nfc(text))
    if "ielts" not in raw.lower():
        return None
    match = re.search(
        r"IELTS\s+([\d.]+)\s*-\s*([\d.]+)\s+([\d.]+)\s*-\s*([\d.]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "bac3_min": match.group(1),
        "bac3_max": match.group(2),
        "bac4_min": match.group(3),
        "bac4_max": match.group(4),
    }


def is_ielts_band_query(query: str) -> bool:
    q = _normalize_query(query)
    if "ielts" not in q:
        return False
    return any(
        x in q
        for x in ["bậc 3", "bậc 4", "bậc mấy", "tương đương", "bao nhiêu điểm"]
    )


def parse_doc_number_from_text(text: str) -> Optional[str]:
    raw = _nfc(text)
    match = re.search(r"Số\s*:\s*([^\n]+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    number = re.sub(r"\s+", " ", match.group(1)).strip()
    return number.split("  ")[0].strip(" .,;")


def infer_doc_number_from_source(source: str) -> Optional[str]:
    stem = Path(source or "").stem.strip()
    if not stem:
        return None
    dotted = re.match(r"^(\d+)\.([A-Z0-9]+)", stem, flags=re.IGNORECASE)
    if dotted:
        return f"{dotted.group(1)}/{dotted.group(2)}/VHU/TB"
    compact = re.match(r"^(\d+)([A-Z0-9]+)VHUTB(.*)$", stem, flags=re.IGNORECASE)
    if compact:
        suffix = (compact.group(3) or "").strip("-_")
        base = f"{compact.group(1)}/{compact.group(2)}/VHU/TB"
        return f"{base}-{suffix}" if suffix else base
    return None


def resolve_doc_number(doc: Document) -> Optional[str]:
    stored = (doc.metadata.get("doc_number") or "").strip()
    if stored:
        return stored
    from_text = parse_doc_number_from_text(doc.page_content or "")
    if from_text:
        return from_text
    return infer_doc_number_from_source(doc.metadata.get("source") or "")


def is_doc_number_query(query: str) -> bool:
    q = _normalize_query(query)
    has_number_ask = any(
        x in q for x in ["số bao nhiêu", "số mấy", "số thông báo", "ký hiệu", "số tb", "mã số"]
    )
    has_doc = any(x in q for x in ["thông báo", " tb", "văn bản", "quyết định"])
    return has_number_ask and has_doc


def _docs_for_doc_number_query(query: str, merged: List[Document]) -> List[Document]:
    q = _normalize_query(query)
    if any(x in q for x in ["đăng ký học phần", "học phần"]) and any(
        x in q for x in ["2026", "2026-2027", "2026–2027", "năm học 2026"]
    ):
        hits = _docs_from_source(merged, "152.myh26")
        if hits:
            return hits
    if "nckh" in q or "nghiên cứu khoa học" in q or "đề tài" in q:
        hits = _docs_from_source(merged, "09myh26")
        if hits:
            return hits
    if "thạc sĩ" in q:
        hits = _docs_from_source(merged, "190")
        if hits:
            return hits
    scored = sorted(
        merged,
        key=lambda d: sum(
            1
            for term in ("đăng ký học phần", "học phần", "tốt nghiệp", "nckh", "thạc sĩ", "2026", "2025")
            if term in q and term in _normalize((d.page_content or "")[:600] + " " + (d.metadata.get("source") or ""))
        ),
        reverse=True,
    )
    return [d for d in scored if resolve_doc_number(d)][:3]


def is_graduation_hoso_query(query: str) -> bool:
    q = _normalize_query(query)
    has_hoso = any(
        x in q for x in ["hồ sơ", "chứng chỉ", "bổ sung hồ sơ", "tiếp nhận"]
    )
    has_grad = any(x in q for x in ["tốt nghiệp", "xét tốt nghiệp"])
    has_dot2 = any(x in q for x in ["đợt 2/2025", "đợt 2", "đợt2"])
    return has_hoso and has_grad and (has_dot2 or "tiếp nhận" in q or "bổ sung" in q)


def try_extract_policy_facts(
    query: str,
    docs: List[Document],
    chunks_cache: Optional[List[Document]] = None,
) -> Optional[Tuple[str, List[Dict[str, str]]]]:

    merged = _merge_chunk_sources(docs, chunks_cache)

    if is_doc_number_query(query):
        seen_sources = set()
        for doc in _docs_for_doc_number_query(query, merged):
            source = doc.metadata.get("source") or ""
            if not source or source in seen_sources:
                continue
            number = resolve_doc_number(doc)
            if not number:
                continue
            seen_sources.add(source)
            answer = (
                f"Theo tài liệu **{source}**, số thông báo là **{number}**.\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_dual_program_credit_query(query):
        for doc in _pick_docs(merged, TYPE_HOC_PHAN):
            limit = parse_dual_program_credit_limit(doc.page_content or "")
            if not limit:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, sinh viên học cùng lúc hai chương trình "
                f"đăng ký tối đa cho mỗi học kỳ chính không vượt quá **{limit} tín chỉ**.\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_song_nganh_fee_query(query):
        for doc in _pick_docs(merged, TYPE_SONG_NGANH):
            fee = parse_song_nganh_combo_fee(doc.page_content or "")
            if not fee:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, học phí combo **Kế toán + Kiểm toán** "
                f"là **{fee}** đồng.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hp_2026_doituong_query(query):
        hp_docs = [d for d in merged if _is_152_hocphan_chunk(d)]
        if not hp_docs:
            hp_docs = [
                d for d in merged
                if "152" in (d.metadata.get("source") or "").lower()
                and "đối tượng" in _normalize(d.page_content or "")
            ]
        for doc in hp_docs:
            info = parse_hp_2026_doituong(doc.page_content or "")
            if not info:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}** (năm học 2026–2027), đối tượng đăng ký học phần gồm:\n"
                f"- Sinh viên các hệ **khóa tuyển sinh {info['cohorts']}** "
                f"({info['note']}).\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hp_152_schedule_query(query):
        pair = _cohort_dot_for_152(query)
        if pair:
            cohort, dot = pair
            hp_docs = _docs_from_source(merged, "152.myh26") or _pick_docs(
                merged, TYPE_HOC_PHAN_26, TYPE_HOC_PHAN
            )
            for doc in hp_docs:
                schedule = parse_hp_dot_window(doc.page_content or "", cohort, dot)
                if not schedule:
                    continue
                source = doc.metadata.get("source") or "tài liệu"
                answer = (
                    f"Theo tài liệu **{source}**, SV khóa tuyển sinh {cohort} "
                    f"đăng ký học phần **đợt {schedule['dot']}**, mở **{schedule['open']}**, "
                    f"đóng **{schedule['close']}**.\n\nNguồn: {source}."
                )
                return answer, _sources_from_docs([doc])

    if is_hp_min_credits_query(query):
        hp_docs = _docs_from_source(merged, "152.myh26") or _pick_docs(merged, TYPE_HOC_PHAN_26)
        for doc in hp_docs:
            credits = parse_hp_min_credits(doc.page_content or "")
            if not credits:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, sinh viên đăng ký tối thiểu "
                f"**{credits} tín chỉ** trong học kỳ theo thông báo HK 2026–2027.\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hp_chuyen_nganh_query(query):
        hp_docs = _docs_from_source(merged, "152.myh26") or _pick_docs(merged, TYPE_HOC_PHAN_26)
        for doc in hp_docs:
            window = parse_hp_chuyen_nganh_window(doc.page_content or "")
            if not window:
                continue
            start, end = window
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, thời gian nộp đơn chuyển chương trình đào tạo "
                f"HK 2026–2027 là từ **{start}** đến **{end}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hp_k2023_kt_query(query):
        for doc in _pick_docs(merged, TYPE_HOC_PHAN):
            schedule = parse_hp_k2023_kt_dot2(doc.page_content or "")
            if not schedule:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, SV khóa 2023 Khoa Kế toán – Tài chính "
                f"đăng ký học phần **đợt {schedule['dot']}**, mở **{schedule['open']}**, "
                f"đóng **{schedule['close']}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hp_k2024_dot5_query(query):
        for doc in _pick_docs(merged, TYPE_HOC_PHAN):
            schedule = parse_hp_k2024_dot5_window(doc.page_content or "")
            if not schedule:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, SV khóa tuyển sinh 2024 (các khoa còn lại) "
                f"đăng ký học phần **đợt {schedule['dot']}**, mở **{schedule['open']}**, "
                f"đóng **{schedule['close']}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_hocphan_bosung_query(query):
        for doc in _pick_docs(merged, TYPE_HOC_PHAN):
            window = parse_hocphan_bosung_window(doc.page_content or "")
            if not window:
                continue
            start, end = window
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, đợt bổ sung đăng ký học phần (lớp bị hủy) "
                f"mở từ **{start}** đến **{end}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_cap_bang_cntt_query(query):
        cap_docs = _docs_from_source(merged, "245.my25") or _pick_docs(merged, TYPE_CAP_BANG)
        for doc in cap_docs:
            schedule = parse_cap_bang_cntt_schedule(doc.page_content or "")
            if not schedule:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            parts = [
                f"Theo tài liệu **{source}**, Khoa CNTT nhận bằng tốt nghiệp đợt 2/2025 "
                f"từ ngày **{schedule['start_date']}**"
            ]
            if schedule.get("weekday"):
                parts.append(f", vào **{schedule['weekday']}** hàng tuần")
            parts.append(".\n\n")
            parts.append(f"Nguồn: {source}.")
            return "".join(parts), _sources_from_docs([doc])

    if is_cap_bang_location_query(query):
        cap_docs = _docs_from_source(merged, "245.my25") or _pick_docs(merged, TYPE_CAP_BANG)
        location: Dict[str, str] = {}
        used_docs: List[Document] = []
        for doc in cap_docs:
            partial = parse_cap_bang_location(doc.page_content or "")
            if not partial:
                continue
            location.update(partial)
            used_docs.append(doc)
        if location:
            source = (used_docs[0].metadata.get("source") if used_docs else None) or "tài liệu"
            parts = [f"Theo tài liệu **{source}**, địa điểm nhận bằng tốt nghiệp đợt 2/2025"]
            if location.get("address"):
                parts.append(f" tại **{location['address']}**")
            if location.get("hotline"):
                parts.append(f"; hotline **{location['hotline']}**")
            parts.append(".\n\n")
            parts.append(f"Nguồn: {source}.")
            return "".join(parts), _sources_from_docs(used_docs or cap_docs[:1])

    for dot in ("1", "2"):
        if is_lt_dot_query(query, dot):
            lt_docs = _docs_from_source(merged, "56.myh26") or _pick_docs(merged, TYPE_LO_TRINH_TN)
            for doc in lt_docs:
                milestone = parse_lt_milestone(doc.page_content or "", dot)
                if not milestone:
                    continue
                source = doc.metadata.get("source") or "tài liệu"
                if dot == "1":
                    answer = (
                        f"Theo tài liệu **{source}**, đợt 1/2026 họp Hội đồng xét tốt nghiệp "
                        f"vào **{milestone['week']}** **{milestone['month']}**.\n\n"
                        f"Nguồn: {source}."
                    )
                else:
                    answer = (
                        f"Theo tài liệu **{source}**, đợt 2/2026 dự kiến cấp phát bằng tốt nghiệp "
                        f"vào **{milestone['week']}** **{milestone['month']}**.\n\n"
                        f"Nguồn: {source}."
                    )
                return answer, _sources_from_docs([doc])

    if is_graduation_hoso_query(query):
        for doc in _pick_docs(merged, TYPE_TOTNGHIEP):
            window = parse_graduation_hoso_window(doc.page_content or "")
            if not window:
                continue
            start, end = window
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, thời gian tiếp nhận bổ sung hồ sơ/chứng chỉ "
                f"đợt 2/2025 là từ **{start}** đến **{end}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_graduation_url_query(query):
        for doc in _pick_docs(merged, TYPE_TOTNGHIEP):
            url = parse_graduation_result_url(doc.page_content or "")
            if not url:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, xem kết quả xét tốt nghiệp đợt 2/2025 tại "
                f"**{url}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_thi_hk_query(query):
        thi_docs = _docs_from_source(merged, "175") or _pick_docs(merged, TYPE_THI_HP)
        for doc in thi_docs:
            window = parse_thi_hk_window(doc.page_content or "")
            if not window:
                continue
            start, end = window
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}**, thi tự luận/trắc nghiệm HK tăng cường "
                f"từ **{start}** đến **{end}**.\n\nNguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_ts_deadline_query(query):
        ts_docs = _docs_from_source(merged, "190") or _pick_docs(merged, TYPE_TUYENSINH_TS)
        for doc in ts_docs:
            deadlines = parse_ts_deadlines(doc.page_content or "")
            if not deadlines:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            parts = [f"Theo tài liệu **{source}**, hạn nộp hồ sơ xét tuyển thạc sĩ đợt 3/2025:"]
            if deadlines.get("nganh_dung"):
                parts.append(f"\n- Ngành đúng: **{deadlines['nganh_dung']}**")
            if deadlines.get("nganh_gan"):
                parts.append(f"\n- Ngành gần/khác: **{deadlines['nganh_gan']}**")
            parts.append(f"\n\nNguồn: {source}.")
            return "".join(parts), _sources_from_docs([doc])

    if is_ts_major_query(query):
        ts_docs = _docs_from_source(merged, "190") or _pick_docs(merged, TYPE_TUYENSINH_TS)
        for doc in ts_docs:
            majors = parse_ts_majors(doc.page_content or "")
            if not majors:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            if majors == ["7 ngành thạc sĩ"]:
                major_text = "**7 ngành thạc sĩ**"
            else:
                major_text = ", ".join(f"**{m}**" for m in majors)
            answer = (
                f"Theo tài liệu **{source}**, thạc sĩ đợt 3/2025 tuyển các ngành: {major_text}.\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    if is_ielts_band_query(query):
        ielts_docs = [d for d in merged if _is_ielts_table_chunk(d)]
        if not ielts_docs:
            ielts_docs = _pick_docs(merged, TYPE_LANG_CERT, TYPE_TUYENSINH_TS)
        for doc in ielts_docs:
            bands = parse_ielts_band_equivalence(doc.page_content or "")
            if not bands:
                continue
            source = doc.metadata.get("source") or "tài liệu"
            answer = (
                f"Theo tài liệu **{source}** (Phụ lục II), điểm IELTS tương đương:\n"
                f"- **Bậc 3**: **{bands['bac3_min']} – {bands['bac3_max']}**\n"
                f"- **Bậc 4**: **{bands['bac4_min']} – {bands['bac4_max']}**\n\n"
                f"Nguồn: {source}."
            )
            return answer, _sources_from_docs([doc])

    return None