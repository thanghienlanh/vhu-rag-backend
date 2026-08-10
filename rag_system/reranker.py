"""
reranker.py
Simple but effective cosine-similarity reranker (no extra heavy models).
Uses the same embeddings to score query vs retrieved chunks.
"""

from typing import Dict, List, Optional

import re
import unicodedata

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval_rules import (
    TYPE_CAP_BANG,
    TYPE_HOC_PHAN,
    TYPE_NCKH_CNTT,
    TYPE_SONG_NGANH,
    TYPE_THI_HP,
    TYPE_TOTNGHIEP,
    chunk_has_type,
    filter_docs_by_topic,
    ielts_cert_sort_key,
    keyword_boost_for_doc,
    nckh_lecturer_sort_key,
)


def _query_intent(query: str) -> Dict[str, bool]:
    q = (query or "").lower()
    wants_graduation = any(
        x in q for x in ["tốt nghiệp", "xét tốt nghiệp", "hồ sơ xét", "cấp bằng", "nhận bằng", "làm bằng"]
    ) or (
        ("hồ sơ" in q or "chứng chỉ" in q)
        and any(x in q for x in ["đợt 2", "xét tốt nghiệp", "tốt nghiệp", "tiếp nhận"])
        and "học phần" not in q
        and "lớp học phần" not in q
    )
    wants_hocphan = any(
        x in q
        for x in [
            "học phần", "đăng ký học phần", "lớp học phần", "bị hủy",
            "khóa tuyển sinh", "song ngành", "tín chỉ",
        ]
    )
    if not wants_hocphan and not wants_graduation and "đợt" in q:
        wants_hocphan = any(x in q for x in ["học phần", "lớp học phần", "đăng ký học phần", "portal", "mở:", "đóng:"])
    if not wants_hocphan and "bổ sung" in q and not wants_graduation:
        wants_hocphan = any(x in q for x in ["học phần", "lớp", "đăng ký"]) and "hồ sơ" not in q
    if wants_graduation:
        wants_hocphan = wants_hocphan and any(x in q for x in ["học phần", "đăng ký học phần", "lớp học phần", "bị hủy"])
    wants_nckh = any(x in q for x in ["đề tài", "nckh", "nghiên cứu khoa học", "đăng ký đề tài", "thời hạn đăng ký"])
    wants_tuyensinh = any(
        x in q for x in [
            "tuyển sinh", "xét tuyển", "nhập học", "tốt nghiệp", "cấp bằng",
            "ielts", "toefl", "chứng chỉ ngoại ngữ", "ngoại ngữ", "bậc 3",
            "bậc 4", "hồ sơ xét",
        ]
    )
    return {
        "hocphan": wants_hocphan,
        "nckh": wants_nckh,
        "tuyensinh": wants_tuyensinh,
        "graduation": wants_graduation,
        "yes_no": any(x in q for x in ["có được", "có thể", "được không"]),
    }

def _fold_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFD", (value or "").casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace("?", "d")


def _phrase_anchor_boost(query: str, text: str) -> float:
    """Reward preserved intent phrases and exact numeric anchors without replacing semantic scoring."""
    stop = {"la", "ai", "va", "cua", "cho", "voi", "the", "nhung", "nao", "bao", "nhieu", "duoc", "trong", "mot", "sinh", "vien", "nguoi", "thong", "tin", "tai", "lieu", "theo", "vhu", "hay", "tra", "loi", "minh", "biet", "thoi", "gian", "ngay", "khi"}
    qfold = _fold_ascii(query)
    tfold = _fold_ascii(text)
    tokens = [token for token in re.findall(r"[a-z0-9]+", qfold) if token not in stop]
    phrases = {" ".join(tokens[i:i + width]) for width in (3, 2) for i in range(len(tokens) - width + 1)}
    phrase_hits = sum(phrase in tfold for phrase in phrases)
    numeric_anchors = set(re.findall(r"20\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?", qfold))
    numeric_hits = sum(anchor in tfold for anchor in numeric_anchors)
    return min(0.28, phrase_hits * 0.045) + min(0.16, numeric_hits * 0.05)


def _doc_key(doc: Document) -> tuple:
    """Stable enough key for de-duplicating retrieved chunks."""
    return (
        doc.metadata.get("source"),
        doc.metadata.get("chunk_id"),
        doc.metadata.get("page"),
        (doc.page_content or "")[:80],
    )


def rerank_documents(
    query: str,
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    top_k: int = 5,
    return_scores: bool = False,
) -> List[Document] | List[tuple[Document, float]]:
    """
    Rerank retrieved documents using cosine similarity + keyword boost.

    This helps reduce confusion between similar official notices (e.g. different "đăng ký" docs).

    If return_scores=True, returns list of (doc, score) tuples.
    """
    if not documents:
        return []

    if len(documents) <= top_k and not return_scores:
        return documents

    # Cap embedding batch for large models (e.g. BGE-M3 on CPU)
    max_embed = 10
    try:
        from rag_config import RERANK_MAX_EMBED_DOCS  # type: ignore
        max_embed = int(RERANK_MAX_EMBED_DOCS)
    except Exception:
        pass
    if len(documents) > max_embed:
        documents = documents[:max_embed]

    # Embed query
    query_embedding = np.array(embeddings.embed_query(query))

    # Embed candidate documents
    doc_texts = [doc.page_content for doc in documents]
    doc_embeddings = np.array(embeddings.embed_documents(doc_texts))

    # Cosine similarity (semantic)
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
    semantic_scores = np.dot(doc_norms, query_norm)

    # Add keyword overlap boost (very helpful for distinguishing similar official notices)
    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 2]
    intent = _query_intent(query)
    keyword_scores = []
    for doc in documents:
        text_lower = doc.page_content.lower()
        overlap = sum(1 for kw in keywords if kw in text_lower)
        boost = keyword_boost_for_doc(query, intent, doc)
        boost += _phrase_anchor_boost(query, doc.page_content)
        if "đề tài" in query_lower and "đề tài" in text_lower:
            boost += 0.12
        if "nckh" in query_lower and ("nckh" in text_lower or "nghiên cứu khoa học" in text_lower):
            boost += 0.15
        if "thời hạn đăng ký" in query_lower and "thời hạn đăng ký" in text_lower:
            boost += 0.18

        if "khoa " in query_lower:
            khoa_match = re.search(r'khoa\s+([^\.\,\n]+)', query_lower)
            if khoa_match:
                query_khoa = khoa_match.group(1).strip()
                if query_khoa in text_lower:
                    boost += 0.25
                elif any(k in text_lower for k in ["khoa công nghệ thông tin", "khoa cntt", "khoa kinh tế", "khoa kế toán"]):
                    boost -= 0.20
        keyword_scores.append(overlap * 0.04 + boost)

    combined = semantic_scores + np.array(keyword_scores)

    scored_docs = list(zip(combined, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    if return_scores:
        return scored_docs[:top_k]

    reranked = [doc for _, doc in scored_docs[:top_k]]
    return reranked


def filter_relevant_chunks(
    query: str,
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    min_score: float = 0.22,
    min_chunks: int = 1,
    strict: bool = False,
    preserve_order: bool = False,
) -> List[Document]:
    """
    Relevance Guard: Filter out chunks that are not sufficiently relevant to the query.

    Uses cosine similarity.
    If strict=True and insufficient high-score chunks, return [] (enables forced refusal).
    """
    if not documents:
        return []

    scored = rerank_documents(query, documents, embeddings, top_k=len(documents), return_scores=True)

    relevant = [(score, doc) for score, doc in scored if score >= min_score]

    if len(relevant) < min_chunks:
        if strict:
            relevant = []
        else:
            relevant = scored[:max(min_chunks, 3)]

    if preserve_order:
        relevant_keys = {_doc_key(doc) for _, doc in relevant}
        return [doc for doc in documents if _doc_key(doc) in relevant_keys]

    relevant.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in relevant]


def expand_adjacent_chunks(
    documents: List[Document],
    all_chunks: Optional[List[Document]],
    window: int = 1,
    max_chunks: Optional[int] = None,
) -> List[Document]:
    """
    Add neighboring chunks from the same source using chunk_id metadata.

    Semantic chunking can split a title, conditions, and deadline into adjacent
    chunks. If retrieval finds one of them, the surrounding chunks often contain
    the rest of the answer. This is a generic retrieval fix, not answer logic.
    """
    if not documents or not all_chunks or window <= 0:
        return documents[:max_chunks] if max_chunks else documents

    by_source_and_id = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("source")
        chunk_id = chunk.metadata.get("chunk_id")
        if source is None or chunk_id is None:
            continue
        try:
            by_source_and_id[(source, int(chunk_id))] = chunk
        except (TypeError, ValueError):
            continue

    expanded = []
    seen = set()

    def add(doc: Document):
        content = (doc.page_content or "").strip()
        if len(content) < 40 and not any(ch.isdigit() for ch in content):
            return
        key = _doc_key(doc)
        if key in seen:
            return
        seen.add(key)
        expanded.append(doc)

    for doc in documents:
        source = doc.metadata.get("source")
        chunk_id = doc.metadata.get("chunk_id")
        try:
            center = int(chunk_id)
        except (TypeError, ValueError):
            add(doc)
            continue

        for offset in range(-window, window + 1):
            neighbor = by_source_and_id.get((source, center + offset))
            if neighbor is not None:
                add(neighbor)
            elif offset == 0:
                add(doc)

            if max_chunks and len(expanded) >= max_chunks:
                return expanded

    return expanded[:max_chunks] if max_chunks else expanded


def prefer_tuyensinh_narrative_chunks(documents: List[Document]) -> List[Document]:
    """
    Put THÔNG BÁO headers and narrative chunks before mid-table rows.
    Table fragments ranked first often cause small LLMs to refuse even when the
    announcement body is present in later chunks.
    """
    if not documents:
        return documents

    import re

    def narrative_score(doc: Document) -> int:
        text = (doc.page_content or "").strip()
        text_lower = text.lower()
        score = 0

        if text_lower.startswith("[tài liệu: thông báo"):
            score += 60
        if "thông báo" in text_lower and "tuyển sinh" in text_lower:
            score += 50
        if "trường đại học văn hiến" in text_lower and "thông báo" in text_lower:
            score += 35
        if "đối tượng tuyển sinh" in text_lower or "ngành tuyển sinh" in text_lower:
            score += 25
        if "hội đồng tuyển sinh" in text_lower:
            score += 20

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and re.match(r"^\d+\.", lines[0]):
            score -= 25
        if len(text) < 120 and "xét tuyển" in text_lower:
            score -= 20
        if text.count("Xét tuyển") >= 3 and "thông báo" not in text_lower:
            score -= 30

        return score

    return sorted(documents, key=narrative_score, reverse=True)


def prefer_ielts_cert_chunks(documents: List[Document]) -> List[Document]:
    """Put PHỤ LỤC II IELTS table chunks first for language-cert questions."""
    if not documents:
        return documents
    return sorted(documents, key=ielts_cert_sort_key, reverse=True)


def prefer_nckh_lecturer_chunks(documents: List[Document]) -> List[Document]:
    """Ensure lecturer-table chunks stay in context for NCKH GVHD queries."""
    if not documents:
        return documents
    typed = [doc for doc in documents if nckh_lecturer_sort_key(doc) >= 30]
    if typed:
        typed_sorted = sorted(typed, key=nckh_lecturer_sort_key, reverse=True)
        top_source = typed_sorted[0].metadata.get("source")
        same_source = [doc for doc in typed_sorted if doc.metadata.get("source") == top_source]
        if same_source:
            return same_source
        return typed_sorted
    return sorted(documents, key=nckh_lecturer_sort_key, reverse=True)


def prefer_primary_source(question: str, documents: List[Document], min_chunks: int = 2) -> List[Document]:
    """
    For direct lookup questions, keep the dominant source once it has enough
    evidence. Broad/comparison questions are allowed to keep multiple sources.
    """
    if not documents:
        return documents

    q = (question or "").lower()
    multi_source_markers = [
        "so sánh",
        "khác nhau",
        "giống nhau",
        "phân biệt",
        "đồng thời",
        "vừa",
        "cả hai",
        "các văn bản",
        "tất cả tài liệu",
        "những tài liệu",
        "trong thư mục",
    ]
    if any(marker in q for marker in multi_source_markers):
        return documents

    wants_hocphan = any(
        x in q
        for x in ["học phần", "đăng ký học phần", "đợt", "khóa tuyển sinh", "song ngành", "tín chỉ", "bổ sung"]
    )
    wants_nckh = any(x in q for x in ["đề tài", "nckh", "nghiên cứu khoa học", "đăng ký đề tài"])
    wants_totnghiep = any(x in q for x in ["tốt nghiệp", "xét tốt nghiệp", "hồ sơ xét", "hồ sơ xét tốt nghiệp"])
    if wants_totnghiep:
        grad_docs = filter_docs_by_topic(
            documents,
            TYPE_TOTNGHIEP,
        )
        grad_docs = [
            doc for doc in grad_docs
            if any(x in (doc.page_content or "").lower() for x in ["tốt nghiệp", "đăng ký xét", "đợt 2"])
        ]
        if grad_docs:
            return grad_docs
    if wants_hocphan and not wants_nckh and not wants_totnghiep:
        schedule_docs = filter_docs_by_topic(documents, TYPE_HOC_PHAN)
        schedule_docs = [
            doc for doc in schedule_docs
            if "đợt" in (doc.page_content or "").lower()
            and "mở" in (doc.page_content or "").lower()
        ]
        if schedule_docs:
            return schedule_docs

    top_source = documents[0].metadata.get("source")
    if not top_source:
        return documents

    primary = [doc for doc in documents if doc.metadata.get("source") == top_source]
    return primary if len(primary) >= min_chunks else documents


def light_keyword_boost_reorder(query: str, documents: List[Document], top_k: int = None) -> List[Document]:
    """
    Cheap, always-on protector (no embeddings): boost documents that strongly match
    critical distinguishing phrases and faculty names in queries about registration.
    Use this in fast_mode and non-fast to prevent mixing NCKH vs học phần and wrong Khoa.
    """
    if not documents:
        return documents
    if top_k is None:
        top_k = len(documents)

    q = query.lower()
    has_khoa = "khoa " in q
    intent = _query_intent(query)
    wants_nckh = intent["nckh"]
    wants_hocphan = intent["hocphan"]
    wants_daihoc = "đại học" in q
    wants_tuyensinh = intent["tuyensinh"]
    query_khoa = None
    import re
    m = re.search(r'khoa\s+([a-zA-ZÀ-ỹ\s\-–]+)', q)
    if m:
        query_khoa = m.group(1).strip().lower()[:40]

    scored = []
    for d in documents:
        text = (d.page_content or "").lower()
        boost = keyword_boost_for_doc(query, intent, d)

        both_intent = wants_nckh and wants_hocphan
        nckh_boost = 0.22 if both_intent else 0.35
        hocphan_boost = 0.22 if both_intent else 0.35
        if wants_nckh and ("đề tài nghiên cứu khoa học" in text or "đăng ký đề tài" in text or "nckh" in text):
            boost += nckh_boost
        if wants_hocphan and ("đăng ký học phần" in text or "kế hoạch đăng ký học phần" in text or "đợt 1" in text or "mở: 10h00" in text or "đợt 2" in text or ('đợt' in text and 'mở' in text and 'đóng' in text)):
            boost += hocphan_boost + 0.4
        if wants_hocphan and chunk_has_type(d, TYPE_HOC_PHAN) and "song ngành" in text and "tín chỉ" in q:
            boost += 0.55
        if wants_hocphan and "bổ sung" in q and "bổ sung" in text:
            boost += 0.65
        if wants_tuyensinh and "ielts" in q and "ielts" in text and "phụ lục ii" in text:
            boost += 0.75
        if wants_tuyensinh and "tốt nghiệp" in q and chunk_has_type(d, TYPE_TOTNGHIEP) and "đợt 2" in text:
            boost += 0.70
        if wants_nckh and "thời hạn đăng ký" in text:
            boost += 0.80
        if 'kế hoạch đăng ký học phần' in text:
            boost += 0.3

        pure_hocphan = wants_hocphan and not wants_nckh
        pure_nckh = wants_nckh and not wants_hocphan
        if pure_hocphan and chunk_has_type(d, TYPE_NCKH_CNTT):
            boost -= 0.55
        if pure_hocphan and "đợt" in q and "kế hoạch đăng ký học phần" in text and chunk_has_type(d, TYPE_HOC_PHAN):
            boost += 0.70
        if pure_nckh and chunk_has_type(d, TYPE_HOC_PHAN):
            boost -= 0.30

        if has_khoa and query_khoa:
            if query_khoa in text or (query_khoa in "kinh tế" and ("kinh tế" in text or "kinh tế - quản trị" in text)):
                boost += 0.45
            elif any(kw in text for kw in ["khoa công nghệ thông tin", "khoa cntt", "khoa kinh tế", "khoa kế toán"]):
                if not (query_khoa in text):
                    boost -= 0.25

        if ("danh sách giảng viên" in text or ("stt" in text and "họ tên" in text)):
            if "giảng viên" in q or "danh sách" in q or "hướng dẫn" in q:
                boost += 0.65
        elif ("giảng viên" in text or "danh sách" in text) and "hướng dẫn" in text:
            if "đề tài" in q or "nckh" in q or "nghiên cứu khoa học" in q:
                boost += 0.40

        if any(x in q for x in ["năm 1", "năm thứ", "đối tượng", "được đăng ký", "có được"]):
            if "đối tượng đăng ký" in text or "năm thứ 2" in text or "năm thứ 3" in text:
                boost += 0.45

        if wants_tuyensinh and not wants_nckh and not wants_hocphan:
            if any(x in text for x in ["tuyển sinh", "xét tuyển", "ngành tuyển sinh", "điều kiện tuyển sinh"]):
                boost += 0.55
            if "thông báo" in text and "tuyển sinh" in text:
                boost += 0.45
            if wants_daihoc and "đại học" in text and "nhập học" in text:
                boost += 0.55
            if "thi kết thúc học phần" in text or "thi kết thúc" in text:
                boost -= 0.70

        if wants_nckh and any(x in q for x in ["lĩnh vực", "nội dung nghiên cứu", "tính chất"]):
            if "lĩnh vực" in text and any(x in text for x in ["công nghệ thông tin", "khoa học máy tính", "thiết kế đồ họa"]):
                boost += 0.60
            if "tuyển sinh" in text and not chunk_has_type(d, TYPE_NCKH_CNTT):
                boost -= 0.65

        scored.append((boost, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]