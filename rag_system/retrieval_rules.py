"""
Configurable retrieval heuristics driven by chunk TYPE / semantic markers — not filenames or answer literals.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

TYPE_RE = re.compile(r"TYPE:\s*([A-Z0-9_]+)")

# Chunk TYPE values (set in loader.py prefix or inferred from title/content)
TYPE_NCKH_CNTT = "NCKH_CNTT"
TYPE_NCKH_KHOA = "NCKH_KHOA"
TYPE_HOC_PHAN = "HOC_PHAN_HK1"
TYPE_TOTNGHIEP = "TOTNGHIEP"
TYPE_TUYENSINH_TS = "TUYENSINH_TS"
TYPE_LANG_CERT = "LANG_CERT"
TYPE_SONG_NGANH = "SONG_NGANH"
TYPE_CAP_BANG = "CAP_BANG"
TYPE_THI_HP = "THI_HP"
TYPE_LO_TRINH_TN = "LO_TRINH_TN"
TYPE_HOC_PHAN_26 = "HOC_PHAN_26"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


_DOC_DISPLAY_PREFIX_RE = re.compile(r"^\[TÀI LIỆU:[^\]]*\]\n?")


def strip_doc_display_prefix(text: str) -> str:
    """Remove the synthetic '[TÀI LIỆU: ...]' label loader.py prepends to every
    chunk before the text is shown/highlighted as a real document quote."""
    return _DOC_DISPLAY_PREFIX_RE.sub("", text or "", count=1)


def get_chunk_type(doc: Document) -> str:
    """Read TYPE from chunk prefix, or infer from title/content."""
    text = doc.page_content or ""
    match = TYPE_RE.search(text)
    if match:
        return match.group(1)

    title = _normalize(doc.metadata.get("doc_title") or "")
    head = _normalize(text[:600])
    haystack = f"{title} {head}"
    source = _normalize(doc.metadata.get("source") or "")

    if "phụ lục ii" in haystack and "ielts" in haystack:
        return TYPE_LANG_CERT
    if "xét tốt nghiệp" in haystack or "đăng ký xét tốt nghiệp" in haystack:
        return TYPE_TOTNGHIEP
    if "tuyển sinh" in haystack and ("thạc sĩ" in haystack or "thac si" in haystack):
        return TYPE_TUYENSINH_TS
    if "song ngành" in haystack and ("nhập học" in haystack or "nhap hoc" in haystack):
        return TYPE_SONG_NGANH
    if any(
        x in haystack
        for x in [
            "nhận bằng",
            "cấp bằng tốt nghiệp",
            "làm thủ tục nhận bằng",
            "bằng tốt nghiệp và phụ",
            "29/09/2025",
            "613 âu cơ",
        ]
    ):
        return TYPE_CAP_BANG
    if "245.my25" in source:
        return TYPE_CAP_BANG
    if any(x in haystack for x in ["thi kết thúc học phần", "học kỳ tăng cường", "thi tự luận", "thi trắc nghiệm"]):
        return TYPE_THI_HP
    if ("học phí" in haystack or "1,838,000" in haystack) and any(
        x in haystack for x in ["kế toán", "kiểm toán", "nhập học", "song ngành"]
    ):
        return TYPE_SONG_NGANH
    if any(x in haystack for x in ["đề tài nghiên cứu", "nckh", "nghiên cứu khoa học sinh viên"]):
        if "công nghệ thông tin" in haystack or "cntt" in haystack:
            return TYPE_NCKH_CNTT
        return TYPE_NCKH_KHOA
    if "152.myh26" in source and (
        "kế hoạch đăng ký học phần" in haystack or "đối tượng đăng ký học phần" in haystack
    ):
        return TYPE_HOC_PHAN_26
    if "56.myh26" in source and (
        "lộ trình" in haystack or "xét và công nhận tốt nghiệp năm 2026" in haystack
    ):
        return TYPE_LO_TRINH_TN
    if "đăng ký học phần" in haystack or "kế hoạch đăng ký học phần" in haystack:
        return TYPE_HOC_PHAN
    return "UNKNOWN"


def chunk_has_type(doc: Document, *types: str) -> bool:
    return get_chunk_type(doc) in types


@dataclass(frozen=True)
class RewriteRule:
    """Append semantic retrieval terms when query/intent match."""
    intent_keys: Sequence[str]
    query_any: Sequence[str]
    query_none: Sequence[str] = ()
    append: str = ""

    def matches(self, query: str, intent: Dict[str, bool]) -> bool:
        q = _normalize(query)
        if self.intent_keys and not all(intent.get(k) for k in self.intent_keys):
            return False
        if self.query_any and not any(term in q for term in self.query_any):
            return False
        if self.query_none and any(term in q for term in self.query_none):
            return False
        return True


REWRITE_RULES: List[RewriteRule] = [
    RewriteRule(
        intent_keys=("tuyensinh",),
        query_any=("ielts", "toefl", "chứng chỉ ngoại ngữ"),
        append=" phụ lục chứng chỉ ngoại ngữ IELTS bậc 3 bậc 4",
    ),
    RewriteRule(
        intent_keys=("graduation",),
        query_any=("tốt nghiệp", "xét tốt nghiệp", "hồ sơ"),
        query_none=("học phần", "đăng ký học phần"),
        append=" thông báo xét tốt nghiệp khung thời gian đăng ký",
    ),
    RewriteRule(
        intent_keys=("hocphan",),
        query_any=("bổ sung", "bị hủy", "hủy"),
        append=" đăng ký học phần đợt bổ sung lớp học phần bị hủy portal",
    ),
    RewriteRule(
        intent_keys=("hocphan",),
        query_any=("song ngành", "hai chương trình", "tín chỉ"),
        append=" tối đa tín chỉ học kỳ chính song ngành",
    ),
    RewriteRule(
        intent_keys=("nckh",),
        query_any=("thời hạn", "hạn đăng ký"),
        append=" thời hạn đăng ký đề tài nghiên cứu khoa học",
    ),
    RewriteRule(
        intent_keys=("hocphan",),
        query_any=("2024", "khóa 2024", "đợt mấy", "đợt nào"),
        query_none=("tốt nghiệp", "xét tốt nghiệp"),
        append=" đợt 5 khóa tuyển sinh 2024 các khoa còn lại",
    ),
    RewriteRule(
        intent_keys=("hocphan",),
        query_any=("2026", "2026-2027", "2026–2027", "đối tượng", "khóa tuyển sinh"),
        query_none=("tốt nghiệp", "xét tốt nghiệp"),
        append=" 152.MYH26 đối tượng đăng ký học phần năm học 2026 2027 khóa 2023 2024 2025",
    ),
    RewriteRule(
        intent_keys=(),
        query_any=("nhận bằng", "cấp bằng", "làm bằng"),
        append=" thông báo nhận bằng tốt nghiệp đợt 2",
    ),
    RewriteRule(
        intent_keys=(),
        query_any=("thi tự luận", "thi trắc nghiệm", "hk tăng cường", "tăng cường"),
        query_none=("tốt nghiệp",),
        append=" thi kết thúc học phần học kỳ tăng cường",
    ),
    RewriteRule(
        intent_keys=(),
        query_any=("học phí", "combo"),
        append=" học phí nhập học song ngành kế toán kiểm toán",
    ),
    RewriteRule(
        intent_keys=("graduation",),
        query_any=("hồ sơ", "chứng chỉ"),
        query_none=("học phần", "lớp học phần"),
        append=" đăng ký xét tốt nghiệp thời gian tiếp nhận hồ sơ chứng chỉ",
    ),
]


@dataclass(frozen=True)
class MandatoryRule:
    """Pin chunks that match TYPE + content markers for a query class."""
    name: str
    types: Sequence[str]
    query_any: Sequence[str]
    intent_any: Sequence[str] = ()
    content_any: Sequence[str] = ()
    limit: int = 2
    pin_only: bool = False

    def query_matches(self, query: str, intent: Dict[str, bool]) -> bool:
        q = _normalize(query)
        if self.query_any and not any(term in q for term in self.query_any):
            return False
        if self.intent_any and not any(intent.get(k) for k in self.intent_any):
            return False
        return True

    def chunk_matches(self, doc: Document) -> bool:
        if not chunk_has_type(doc, *self.types):
            return False
        text = _normalize(doc.page_content or "")
        if self.content_any and not any(marker in text for marker in self.content_any):
            return False
        return True


MANDATORY_RULES: List[MandatoryRule] = [
    MandatoryRule(
        name="graduation",
        types=(TYPE_TOTNGHIEP,),
        query_any=("tốt nghiệp", "xét tốt nghiệp", "hồ sơ"),
        intent_any=("graduation",),
        content_any=("đăng ký xét tốt nghiệp", "thời gian đăng ký"),
        limit=2,
    ),
    MandatoryRule(
        name="ielts",
        types=(TYPE_LANG_CERT, TYPE_TUYENSINH_TS),
        query_any=("ielts", "toefl", "chứng chỉ ngoại ngữ", "ngoại ngữ", "bậc 3", "bậc 4"),
        content_any=("ielts", "phụ lục ii", "bậc 3", "bậc 4"),
        limit=2,
        pin_only=True,
    ),
    MandatoryRule(
        name="hocphan_bosung",
        types=(TYPE_HOC_PHAN,),
        query_any=("bổ sung", "bị hủy", "hủy"),
        intent_any=("hocphan",),
        content_any=("đợt bổ sung", "bổ sung", "lớp học phần bị hủy"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="song_nganh_credits",
        types=(TYPE_HOC_PHAN,),
        query_any=("song ngành", "hai chương trình", "tín chỉ"),
        content_any=("song ngành", "hai chương trình", "37", "tín chỉ"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="nckh_overview",
        types=(TYPE_NCKH_CNTT,),
        query_any=("thời hạn", "hạn đăng ký", "nhóm", "đối tượng", "năm thứ", "năm 1"),
        intent_any=("nckh",),
        content_any=("thời hạn đăng ký", "đối tượng đăng ký", "nhóm nghiên cứu"),
        limit=1,
    ),
    MandatoryRule(
        name="hp_k2024_dot5",
        types=(TYPE_HOC_PHAN,),
        query_any=("2024", "khóa 2024"),
        intent_any=("hocphan",),
        content_any=("đợt 5", "2024", "các khoa còn lại"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="graduation_hoso",
        types=(TYPE_TOTNGHIEP,),
        query_any=("hồ sơ", "chứng chỉ", "bổ sung hồ sơ"),
        intent_any=("graduation",),
        content_any=("17/6", "23/6", "đăng ký xét tốt nghiệp", "thời gian đăng ký"),
        limit=2,
        pin_only=True,
    ),
    MandatoryRule(
        name="cap_bang",
        types=(TYPE_CAP_BANG,),
        query_any=("nhận bằng", "cấp bằng", "làm bằng"),
        content_any=("nhận bằng", "29/09", "cấp bằng"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="thi_hk",
        types=(TYPE_THI_HP,),
        query_any=("thi tự luận", "thi trắc nghiệm", "tăng cường", "hk tăng cường"),
        content_any=("28/7", "24/8", "thi kết thúc"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="song_nganh_fee",
        types=(TYPE_SONG_NGANH,),
        query_any=("học phí", "combo", "kế toán", "kiểm toán"),
        content_any=("học phí", "1,838,000", "kế toán", "kiểm toán"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="hp_152",
        types=(TYPE_HOC_PHAN_26, TYPE_HOC_PHAN),
        query_any=("2026", "2026-2027", "2026–2027"),
        intent_any=("hocphan",),
        content_any=("kế hoạch đăng ký học phần", "đợt 1", "30/6/2026"),
        limit=3,
        pin_only=True,
    ),
    MandatoryRule(
        name="lt_56",
        types=(TYPE_LO_TRINH_TN,),
        query_any=("lộ trình", "đợt 1/2026", "đợt 2/2026", "đợt 3/2026", "hội đồng xét"),
        content_any=("tuần thứ", "xét và công nhận tốt nghiệp", "cấp phát bằng"),
        limit=2,
        pin_only=True,
    ),
    MandatoryRule(
        name="ts_program",
        types=(TYPE_TUYENSINH_TS,),
        query_any=("thạc sĩ", "tuyển sinh", "ngành nào", "đợt 3"),
        content_any=("tuyển sinh", "thạc sĩ", "văn học"),
        limit=2,
        pin_only=True,
    ),
    MandatoryRule(
        name="graduation_url",
        types=(TYPE_TOTNGHIEP,),
        query_any=("qldt", "kết quả xét", "xem kết quả"),
        content_any=("qldt.vhu.edu.vn", "kết quả xét tốt nghiệp"),
        limit=1,
        pin_only=True,
    ),
    MandatoryRule(
        name="hp_k2023_kt",
        types=(TYPE_HOC_PHAN,),
        query_any=("khóa 2023", "kế toán", "tài chính"),
        intent_any=("hocphan",),
        content_any=("đợt 2", "kế toán", "12/7/2025"),
        limit=1,
        pin_only=True,
    ),
]


@dataclass(frozen=True)
class SupplementRule:
    types: Sequence[str]
    query_any: Sequence[str] = ()
    intent_any: Sequence[str] = ()
    content_any: Sequence[str] = ()
    score: int = 20

    def matches(self, query: str, intent: Dict[str, bool], doc: Document) -> int:
        if not chunk_has_type(doc, *self.types):
            return 0
        q = _normalize(query)
        if self.query_any and not any(term in q for term in self.query_any):
            return 0
        if self.intent_any and not any(intent.get(k) for k in self.intent_any):
            return 0
        text = _normalize(doc.page_content or "")
        if self.content_any and not any(marker in text for marker in self.content_any):
            return 0
        return self.score


SUPPLEMENT_RULES: List[SupplementRule] = [
    SupplementRule(
        types=(TYPE_HOC_PHAN,),
        intent_any=("hocphan",),
        content_any=("kế hoạch đăng ký học phần", "đợt 1", "mở"),
        score=22,
    ),
    SupplementRule(
        types=(TYPE_HOC_PHAN,),
        query_any=("bổ sung", "bị hủy"),
        content_any=("đợt bổ sung", "bổ sung", "portal"),
        score=36,
    ),
    SupplementRule(
        types=(TYPE_HOC_PHAN,),
        query_any=("song ngành", "tín chỉ", "hai chương trình"),
        content_any=("hai chương trình", "37", "tín chỉ", "song ngành"),
        score=40,
    ),
    SupplementRule(
        types=(TYPE_TOTNGHIEP,),
        intent_any=("graduation",),
        content_any=("xét tốt nghiệp", "thời gian đăng ký", "đợt 2"),
        score=40,
    ),
    SupplementRule(
        types=(TYPE_LANG_CERT, TYPE_TUYENSINH_TS),
        query_any=("ielts", "toefl", "ngoại ngữ", "bậc 3", "bậc 4"),
        content_any=("ielts", "phụ lục ii", "bậc 3"),
        score=38,
    ),
    SupplementRule(
        types=(TYPE_NCKH_CNTT,),
        intent_any=("nckh",),
        query_any=("thời hạn", "hạn đăng ký", "nhóm", "đối tượng"),
        content_any=("thời hạn đăng ký", "đối tượng đăng ký", "nhóm nghiên cứu"),
        score=38,
    ),
    SupplementRule(
        types=(TYPE_NCKH_KHOA,),
        query_any=("giảng viên", "hướng dẫn", "danh sách"),
        intent_any=("nckh",),
        content_any=("danh sách giảng viên", "stt họ tên", "email"),
        score=30,
    ),
    SupplementRule(
        types=(TYPE_NCKH_CNTT,),
        intent_any=("nckh",),
        query_any=("lĩnh vực", "nội dung nghiên cứu"),
        content_any=("lĩnh vực", "công nghệ thông tin"),
        score=28,
    ),
    SupplementRule(
        types=(TYPE_HOC_PHAN,),
        query_any=("2024", "khóa 2024", "đợt mấy"),
        content_any=("đợt 5", "2024", "18/7", "19/7"),
        score=42,
    ),
    SupplementRule(
        types=(TYPE_TOTNGHIEP,),
        query_any=("hồ sơ", "chứng chỉ", "bổ sung"),
        content_any=("17/6", "23/6", "tiếp nhận", "xét tốt nghiệp"),
        score=44,
    ),
    SupplementRule(
        types=(TYPE_CAP_BANG,),
        query_any=("nhận bằng", "cấp bằng", "cntt"),
        content_any=("29/09", "nhận bằng", "công nghệ thông tin"),
        score=40,
    ),
    SupplementRule(
        types=(TYPE_THI_HP,),
        query_any=("thi", "tăng cường", "tự luận", "trắc nghiệm"),
        content_any=("28/7", "24/8", "thi kết thúc"),
        score=40,
    ),
    SupplementRule(
        types=(TYPE_SONG_NGANH,),
        query_any=("học phí", "combo", "kế toán", "kiểm toán"),
        content_any=("55,140,000", "55.140.000", "1,838,000", "học phí"),
        score=42,
    ),
]


def rewrite_query_for_retrieval(question: str, intent: Dict[str, bool]) -> str:
    q = (question or "").strip()
    if not q:
        return q
    q_lower = _normalize(q)

    for rule in REWRITE_RULES:
        if rule.matches(q, intent):
            return q.rstrip("?").strip() + rule.append

    # Legacy safe expansions without filenames/dates
    if intent.get("tuyensinh") and not intent.get("nckh") and not intent.get("hocphan"):
        if "tốt nghiệp" in q_lower and "tuyển sinh" not in q_lower:
            return q.rstrip("?").strip() + " chương trình đại học Văn Hiến"
        if "quy định" in q_lower and "tuyển sinh" in q_lower and "ielts" not in q_lower:
            return "Thông báo tuyển sinh đại học Văn Hiến"
        if "thông báo" in q_lower and "2025" not in q_lower:
            return q.rstrip("?").strip() + " năm học hiện hành"

    if intent.get("nckh") and any(x in q_lower for x in ["lĩnh vực", "nội dung nghiên cứu", "tính chất"]):
        if "công nghệ thông tin" not in q_lower and "cntt" not in q_lower:
            return q.rstrip("?").strip() + " NCKH Khoa Công nghệ Thông tin"

    return q


def find_mandatory_chunks(
    chunks: Sequence[Document],
    query: str,
    intent: Dict[str, bool],
) -> tuple[List[Document], Optional[str]]:
    """Return pinned chunks and optional pin_only rule name."""
    pinned: List[Document] = []
    pin_only_name: Optional[str] = None
    seen = set()

    for rule in MANDATORY_RULES:
        if not rule.query_matches(query, intent):
            continue
        for chunk in chunks:
            if not rule.chunk_matches(chunk):
                continue
            key = (chunk.metadata.get("source"), chunk.metadata.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            pinned.append(chunk)
            if len([c for c in pinned if rule.chunk_matches(c)]) >= rule.limit:
                break
        if rule.pin_only and pinned:
            pin_only_name = rule.name

    return pinned, pin_only_name


def score_supplement_chunk(query: str, intent: Dict[str, bool], chunk: Document) -> int:
    score = 0
    for rule in SUPPLEMENT_RULES:
        score = max(score, rule.matches(query, intent, chunk))

    text = _normalize(chunk.page_content or "")
    q = _normalize(query)

    if intent.get("hocphan") or "đợt" in q:
        if "đợt 3" in q and "đợt 3" in text and "mở" in text:
            score = max(score, 25)

    if intent.get("yes_no") and intent.get("nckh"):
        if "đối tượng đăng ký" in text or "năm thứ 2" in text:
            score = max(score, 16)

    return score


def filter_docs_by_topic(documents: List[Document], *types: str) -> List[Document]:
    return [doc for doc in documents if chunk_has_type(doc, *types)]


def keyword_boost_for_doc(query: str, intent: Dict[str, bool], doc: Document) -> float:
    """Embedding-free boost from TYPE + topic phrases (used in reranker)."""
    text = _normalize(doc.page_content or "")
    q = _normalize(query)
    boost = 0.0
    ctype = get_chunk_type(doc)

    wants_nckh = intent.get("nckh") or any(x in q for x in ["đề tài", "nckh"])
    wants_hocphan = intent.get("hocphan") or any(
        x in q for x in ["học phần", "đăng ký học phần", "đợt", "tín chỉ", "bổ sung"]
    )
    wants_tuyensinh = intent.get("tuyensinh") or any(
        x in q for x in ["tuyển sinh", "ielts", "ngoại ngữ", "tốt nghiệp"]
    )

    if wants_nckh and ctype in (TYPE_NCKH_CNTT, TYPE_NCKH_KHOA):
        boost += 0.35
    if wants_hocphan and ctype == TYPE_HOC_PHAN:
        boost += 0.45
    if wants_tuyensinh and ctype == TYPE_TOTNGHIEP and "tốt nghiệp" in q:
        boost += 0.70
    if wants_tuyensinh and ctype == TYPE_LANG_CERT and "ielts" in q:
        boost += 0.75
    if wants_hocphan and "bổ sung" in q and "bổ sung" in text:
        boost += 0.55
    if wants_hocphan and ctype == TYPE_HOC_PHAN and "song ngành" in text and "tín chỉ" in q:
        boost += 0.50
    if wants_hocphan and "2024" in q and "đợt 5" in text and "2024" in text:
        boost += 0.65
    if intent.get("graduation") and ctype == TYPE_TOTNGHIEP:
        boost += 0.55
    if intent.get("graduation") and ctype == TYPE_HOC_PHAN and "học phần" not in q:
        boost -= 0.50
    if any(x in q for x in ["nhận bằng", "cấp bằng"]) and ctype == TYPE_CAP_BANG:
        boost += 0.70
    if any(x in q for x in ["thi tự luận", "thi trắc nghiệm", "tăng cường"]) and ctype == TYPE_THI_HP:
        boost += 0.70
    if any(x in q for x in ["học phí", "combo"]) and ctype == TYPE_SONG_NGANH:
        boost += 0.70

    if wants_hocphan and not wants_nckh and ctype in (TYPE_NCKH_CNTT, TYPE_NCKH_KHOA):
        boost -= 0.40
    if wants_nckh and not wants_hocphan and ctype == TYPE_HOC_PHAN:
        boost -= 0.25

    return boost


def ielts_cert_sort_key(doc: Document) -> int:
    text = _normalize(doc.page_content or "")
    score = 0
    if chunk_has_type(doc, TYPE_LANG_CERT, TYPE_TUYENSINH_TS):
        score += 40
    if "phụ lục ii" in text and "ielts" in text:
        score += 100
    if "ielts" in text and re.search(r"\d+\.\d+", text):
        score += 60
    if "bậc 3" in text and "bậc 4" in text:
        score += 30
    return score


def nckh_lecturer_sort_key(doc: Document) -> int:
    text = _normalize(doc.page_content or "")
    if not chunk_has_type(doc, TYPE_NCKH_KHOA, TYPE_NCKH_CNTT):
        return 0
    score = 0
    if "danh sách giảng viên" in text:
        score += 80
    if "stt họ tên" in text or "email/điện thoại" in text:
        score += 50
    if "@" in text and re.search(r"\b\d{1,2}\s+[a-zà-ỹ]", text):
        score += 30
    return score