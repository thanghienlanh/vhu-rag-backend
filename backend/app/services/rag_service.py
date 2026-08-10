import asyncio
import time
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
import json
import httpx
from datetime import datetime, timezone
from services.query_expansion import normalize_query_for_retrieval
from services.pure_list_extraction import (
    classify_answer_shape,
    extract_records_by_chunk,
)
from services.pure_grounding import (
    build_grounding_prompts,
    extract_boolean_fact,
    extract_extractive_fact,
    extract_location_fact,
    parse_grounded_json,
    select_evidence_window,
    validate_grounded_response,
)
from table_store import valid_rows_for_sources

# Prevent legacy Windows console encodings from turning Unicode request logs into API failures.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
RAG_SYSTEM_DIR = PROJECT_ROOT / "rag_system"

if str(RAG_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_SYSTEM_DIR))


def _load_local_env_file():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()

from query_guard import (
    infer_corpus_max_year,
    is_arithmetic_or_trivia_query as _guard_is_arithmetic_or_trivia_query,
    is_offtopic_query as _guard_is_offtopic_query,
    is_unsupported_future_query as _guard_is_unsupported_future_query,
    query_in_corpus_domain as _guard_query_in_corpus_domain,
)

from rag_config import (
    PAPERS_DIR as DEFAULT_PAPERS_DIR,
    FAISS_FULL_PATH,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    USE_HYBRID_SEARCH,
    INITIAL_RETRIEVE_K,
    FINAL_TOP_K,
    USE_RERANKER,
    USE_QUERY_REWRITING,
    USE_RELEVANCE_GUARD,
    MIN_RELEVANCE_SCORE,
    MIN_RELEVANT_CHUNKS,
    MAX_OUTPUT_TOKENS,
    NUM_CTX,
    PURE_RAG_TEMPERATURE,
    PURE_RAG_MAX_OUTPUT_TOKENS,
    PURE_RAG_NUM_CTX,
    NO_INFO_ANSWER,
    NEIGHBOR_CHUNK_WINDOW,
    MAX_CONTEXT_CHUNKS,
    PURE_RAG,
)
from embeddings import get_embeddings
from vectorstore import load_faiss_index, get_retriever, create_faiss_index, index_needs_rebuild, compute_corpus_fingerprint
from loader import load_documents as load_pdfs  # supports multi format now
from chunker import chunk_documents
from hybrid_retriever import get_hybrid_retriever
from reranker import (
    rerank_documents,
    filter_relevant_chunks,
    light_keyword_boost_reorder,
    expand_adjacent_chunks,
    prefer_primary_source,
    prefer_tuyensinh_narrative_chunks,
    prefer_nckh_lecturer_chunks,
    prefer_ielts_cert_chunks,
)
from lecturer_extractor import is_nckh_lecturer_query, try_extract_nckh_lecturers
from nckh_fact_extractor import try_extract_nckh_facts
from policy_fact_extractor import try_extract_policy_facts
from retrieval_rules import (
    rewrite_query_for_retrieval,
    find_mandatory_chunks,
    score_supplement_chunk,
    strip_doc_display_prefix,
)

from rag_chain import build_rag_chain, format_context, build_prompt
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

DATA_DIR = os.getenv("RAG_DATA_DIR", str(DEFAULT_PAPERS_DIR))
INDEX_DIR = os.getenv("RAG_INDEX_DIR", str(FAISS_FULL_PATH))
MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:3b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", str(OLLAMA_TIMEOUT)))
USE_HYBRID = os.getenv("USE_HYBRID", str(USE_HYBRID_SEARCH)).lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "ollama").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
FAST_MODE_MAX_OUTPUT_TOKENS = int(os.getenv("FAST_MODE_MAX_OUTPUT_TOKENS", "256"))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "999"))
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "4"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
MAX_PARALLEL_QUESTIONS = int(os.getenv("MAX_PARALLEL_QUESTIONS", "3"))
RERANK_MAX_EMBED_DOCS = int(os.getenv("RERANK_MAX_EMBED_DOCS", "10"))
PURE_RAG_MODE = os.getenv("PURE_RAG", str(PURE_RAG)).lower() in ("1", "true", "yes")
RAG_AUDIT_LOG = os.getenv("RAG_AUDIT_LOG", str(PROJECT_ROOT / "logs" / "rag_audit.jsonl"))

_initialized = False
_embeddings = None
_vectorstore = None
_chunks_cache = None
_retriever = None
_llm = None


def _write_rag_audit(question: str, rewritten_query: str, docs: List[Document], logd: Dict[str, Any], refused: bool, reason: str):
    """Record retrieval evidence without storing generated answers."""
    try:
        path = Path(RAG_AUDIT_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "pure_rag" if PURE_RAG_MODE else "legacy_rag",
            "question": question,
            "rewritten_query": rewritten_query,
            "refused": refused,
            "reason": reason,
            "technical_status": logd.get("technical_status"),
            "query_rewrite_seconds": logd.get("query_rewrite_seconds", 0.0),
            "keyword_retrieval_seconds": logd.get("keyword_retrieval_seconds", 0.0),
            "vector_retrieval_seconds": logd.get("vector_retrieval_seconds", 0.0),
            "rerank_seconds": logd.get("rerank_seconds", 0.0),
            "context_assembly_seconds": logd.get("timing_ms", {}).get("context", 0.0) / 1000,
            "generation_seconds": logd.get("timing_ms", {}).get("generation", 0.0) / 1000,
            "total_seconds": logd.get("timing_ms", {}).get("total", 0.0) / 1000,
            "evidence": logd,
            "generated_answer": logd.get("generated_answer"),
            "ollama_eval_count": logd.get("ollama_eval_count"),
            "ollama_done_reason": logd.get("ollama_done_reason"),
            "primary_chunks": logd.get("primary_chunks", []),
            "neighbor_candidates": logd.get("neighbor_candidates", []),
            "selected_context_chunks": logd.get("selected_context_chunks", []),
            "generation_debug": logd.get("generation_debug", {}),
            "grounding_attempts": logd.get("grounding_attempts", []),
            "chunks": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "rerank_score": doc.metadata.get("_pure_rerank_score"),
                }
                for doc in docs
            ],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[Audit] write failed: {exc!r}")


def _chunk_audit_record(doc: Document, score: Optional[float] = None, continuity_score: Optional[float] = None) -> Dict[str, Any]:
    record = {
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "chunk_id": doc.metadata.get("chunk_id"),
        "rerank_score": doc.metadata.get("_pure_rerank_score") if score is None else round(float(score), 6),
    }
    if continuity_score is not None:
        record["continuity_score"] = round(float(continuity_score), 6)
    return record


def _neighbor_continuity_score(primary: Document, neighbor: Document, question: str = "") -> float:
    """Generic continuity signal for adjacent prose and extracted table rows."""
    page_distance = abs(float(neighbor.metadata.get("page") or 0) - float(primary.metadata.get("page") or 0))
    score = 0.01 / (1.0 + page_distance)
    primary_text = primary.page_content[:1000]
    neighbor_text = neighbor.page_content[:1200]
    has_table_header = bool(re.search(r"\b(?:stt|no\.?|email|điện\s*thoại)\b", primary_text, re.IGNORECASE))
    starts_table_row = bool(re.search(r"(?:^|\n|\|)\s*\d{1,3}\s+[A-Za-zÀ-ỹ]", neighbor_text))
    folded_question = _fold_evidence_text(question)
    wants_table_continuity = is_nckh_lecturer_query(question) or any(
        marker in folded_question for marker in ("danh sach", "liet ke", "nhung ai")
    )
    if wants_table_continuity and has_table_header and starts_table_row and "@" in neighbor_text:
        score += 0.35
    return score


def _expand_pure_neighbors(question: str, primary_docs: List[Document], max_chunks: int) -> tuple[List[Document], List[Dict[str, Any]]]:
    """Keep primary retrieval results first, then choose same-source neighbours by semantic rerank."""
    primaries = list(primary_docs[:max_chunks])
    if not primaries or not _chunks_cache or len(primaries) >= max_chunks:
        return primaries, []
    by_key = {(str(doc.metadata.get("source", "")), int(doc.metadata.get("chunk_id", -1))): doc for doc in _chunks_cache if doc.metadata.get("chunk_id") is not None}
    candidates: List[tuple[Document, float]] = []
    seen = {(str(doc.metadata.get("source", "")), doc.metadata.get("chunk_id")) for doc in primaries}
    for primary in primaries:
        source = str(primary.metadata.get("source", "")); primary_id = primary.metadata.get("chunk_id")
        if not isinstance(primary_id, int):
            continue
        for offset in (-1, 1):
            neighbor = by_key.get((source, primary_id + offset)); key = (source, primary_id + offset)
            if neighbor is None or key in seen:
                continue
            seen.add(key)
            candidates.append((neighbor, _neighbor_continuity_score(primary, neighbor, question)))
    if not candidates:
        return primaries, []
    candidate_docs = [item[0] for item in candidates]
    scored = rerank_documents(question, candidate_docs, _embeddings, top_k=len(candidate_docs), return_scores=True) if _embeddings is not None else [(0.0, doc) for doc in candidate_docs]
    continuity_by_id = {id(doc): continuity for doc, continuity in candidates}
    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", _fold_evidence_text(question))
        if len(term) > 1 and term not in {"la", "va", "cua", "cho", "voi", "bao", "nhieu", "lau", "sinh", "vien"}
    }
    ranked = []
    for semantic_score, doc in scored:
        continuity = continuity_by_id[id(doc)]
        neighbor_terms = set(re.findall(r"[a-z0-9]+", _fold_evidence_text(doc.page_content)))
        lexical_overlap = len(query_terms.intersection(neighbor_terms))
        lexical_bonus = min(0.40, 0.08 * lexical_overlap)
        combined = float(semantic_score) + continuity + lexical_bonus
        doc.metadata["_pure_neighbor_rerank_score"] = round(float(semantic_score), 6); doc.metadata["_pure_neighbor_combined_score"] = round(combined, 6)
        ranked.append((combined, semantic_score, doc, continuity))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = primaries + [item[2] for item in ranked[:max_chunks - len(primaries)]]; audit = []
    for combined, semantic_score, doc, continuity in ranked:
        record = _chunk_audit_record(doc, semantic_score, continuity); record["lexical_overlap"] = len(query_terms.intersection(set(re.findall(r"[a-z0-9]+", _fold_evidence_text(doc.page_content))))); record["combined_score"] = round(float(combined), 6); record["selected"] = doc in selected; audit.append(record)
    return selected, audit
# Index processing status for UI
_index_status = "ready"  # idle | indexing | ready | error
_index_error = None

# Optionally expose current global for debug
def _get_raw_index_status():
    return _index_status


def _use_gemini() -> bool:
    return LLM_PROVIDER == "gemini" and bool(GEMINI_API_KEY)


def _llm_backend_label() -> str:
    if _use_gemini():
        return f"Gemini API ({GEMINI_MODEL})"
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        return f"Ollama local ({MODEL}) - Gemini key missing"
    return f"Ollama local ({MODEL})"


def _log_llm_backend(mode: str, question: str = ""):
    preview = (question or "").replace("\n", " ").strip()[:90]
    suffix = f" | question={preview!r}" if preview else ""
    print(f"[LLM] mode={mode} | backend={_llm_backend_label()}{suffix}")


_LLM_STREAM_ERROR = (
    "Không thể tạo câu trả lời (Ollama lỗi hoặc timeout). "
    "Chạy `ollama serve` và thử lại."
)


async def _stream_ollama_chain(chain, inputs: dict, label: str = "LLM") -> AsyncGenerator[str, None]:
    """Stream Ollama with a hard timeout so the UI never ends on a blank answer."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + LLM_TIMEOUT
    try:
        async for chunk in chain.astream(inputs):
            if loop.time() > deadline:
                raise TimeoutError(f"{label} stream exceeded {LLM_TIMEOUT}s")
            if chunk:
                yield chunk
    except Exception as exc:
        print(f"[{label}] Ollama stream failed: {exc!r}")
        yield _LLM_STREAM_ERROR


def _detect_query_faculty(query: str) -> Optional[str]:
    q = (query or "").lower()
    if "công nghệ thông tin" in q or "cntt" in q:
        return "cntt"
    if "kế toán" in q or "tài chính" in q:
        return "kế toán"
    if "kinh tế" in q and "quản trị" in q:
        return "kinh_te_quan_tri"
    if "kinh tế" in q:
        return "kinh tế"
    if "quản trị" in q:
        return "quản trị"
    return None


MIN_COMPOUND_QUESTION_LEN = 12

_QUESTION_TAIL_PATTERNS = (
    "gồm những gì",
    "gồm gì",
    "có phải không",
    "như thế nào",
    "được không",
    "là gì",
    "là ai",
    "ở đâu",
    "khi nào",
    "bao nhiêu",
    "ra sao",
    "thế nào",
    "mấy tín chỉ",
    "mấy năm",
    "mấy tín",
    "có không",
)
_QUESTION_TAIL_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in _QUESTION_TAIL_PATTERNS) + r")",
    re.IGNORECASE,
)


def _normalize_compound_input(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\bEdit\b", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\r\n?", "\n", raw)
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.split("\n")]
    return "\n".join(line for line in lines if line)


def _finalize_question_part(part: str) -> str:
    cleaned = _strip_list_prefix(part.strip())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_compound_questions(questions: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for q in questions:
        part = _finalize_question_part(q)
        if len(part) < MIN_COMPOUND_QUESTION_LEN:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return unique


def _split_by_question_marks(text: str) -> Optional[List[str]]:
    if text.count("?") <= 1:
        return None
    parts = re.split(r"(?<=\?)\s+", text)
    questions: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < MIN_COMPOUND_QUESTION_LEN:
            continue
        if not part.endswith("?"):
            part = part.rstrip(".") + "?"
        questions.append(part)
    return questions if len(questions) > 1 else None


def _split_by_newlines(text: str) -> Optional[List[str]]:
    if "\n" not in text:
        return None
    parts = [line.strip() for line in text.split("\n") if line.strip()]
    valid = [p for p in parts if len(p) >= MIN_COMPOUND_QUESTION_LEN]
    return valid if len(valid) > 1 else None


def _strip_list_prefix(part: str) -> str:
    return re.sub(r"^\s*(?:\d+[\.\)]\s+|[-•*]\s+)", "", part).strip()


def _split_by_numbered_list(text: str) -> Optional[List[str]]:
    markers = list(re.finditer(r"(?:^|\n)\s*\d+[\.\)]\s+", text))
    if len(markers) < 2:
        markers = list(re.finditer(r"(?<!\d)\d+[\.\)]\s+", text))
    if len(markers) < 2:
        return None
    parts = re.split(r"(?:^|\n)\s*\d+[\.\)]\s+|(?<!\d)\d+[\.\)]\s+", text)
    valid = [
        _strip_list_prefix(p)
        for p in parts
        if p.strip() and len(_strip_list_prefix(p)) >= MIN_COMPOUND_QUESTION_LEN
    ]
    return valid if len(valid) > 1 else None


def _split_by_bullets(text: str) -> Optional[List[str]]:
    if not re.search(r"(?:^|\n)\s*[-•*]\s+", text):
        return None
    parts = re.split(r"(?:^|\n)\s*[-•*]\s+", text)
    valid = [
        _strip_list_prefix(p)
        for p in parts
        if p.strip() and len(_strip_list_prefix(p)) >= MIN_COMPOUND_QUESTION_LEN
    ]
    return valid if len(valid) > 1 else None


def _split_by_question_tails(text: str) -> Optional[List[str]]:
    matches = list(_QUESTION_TAIL_RE.finditer(text))
    if not matches:
        return None

    questions: List[str] = []
    last_end = 0
    for match in matches:
        end = match.end()
        remainder = text[end:].strip()
        if not remainder or len(remainder) < MIN_COMPOUND_QUESTION_LEN:
            continue
        chunk = text[last_end:end].strip()
        if len(chunk) < MIN_COMPOUND_QUESTION_LEN:
            continue
        questions.append(chunk)
        last_end = end

    if not questions:
        return None

    final = text[last_end:].strip()
    if len(final) >= MIN_COMPOUND_QUESTION_LEN:
        questions.append(final)

    return questions if len(questions) > 1 else None


def split_compound_questions(text: str) -> List[str]:
    """
    Tách một tin nhắn chứa nhiều câu hỏi thành từng câu riêng.
    Ưu tiên: dấu ?, xuống dòng, danh sách đánh số, bullet, cụm hỏi tiếng Việt.
    """
    normalized = _normalize_compound_input(text)
    if not normalized:
        return []

    collapsed = normalized.replace("\n", " ")
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    q_lower = collapsed.lower()

    # Một câu tra cứu có "hai chương trình / song ngành + tín chỉ" — không tách.
    if (
        any(x in q_lower for x in ["hai chương trình", "song ngành"])
        and "tín chỉ" in q_lower
        and collapsed.count("?") <= 1
    ):
        return [collapsed]

    single_terminal_question = collapsed.count("?") <= 1

    for splitter in (
        lambda: _split_by_question_marks(collapsed),
        lambda: _split_by_numbered_list(normalized),
        lambda: _split_by_bullets(normalized),
        lambda: _split_by_newlines(normalized),
        lambda: _split_by_question_tails(collapsed) if not single_terminal_question else None,
    ):
        parts = splitter()
        if not parts:
            continue
        unique = _dedupe_compound_questions(parts)
        if len(unique) > 1:
            return unique

    return [collapsed] if collapsed else []


def _questions_for_pipeline(question: str, fast_mode: bool = False) -> List[str]:
    """Split compound questions for the standard RAG pipeline."""
    q = (question or "").strip()
    if not q:
        return []
    subs = split_compound_questions(q)
    return subs if subs else [q]


def _merge_sources(source_lists: List[List[Dict]], limit: int = 5) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for sources in source_lists:
        for item in sources:
            key = f"{item.get('source')}|{item.get('page')}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


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
            "học phần",
            "đăng ký học phần",
            "lớp học phần",
            "bị hủy",
            "khóa tuyển sinh",
            "song ngành",
            "tín chỉ",
        ]
    )
    if not wants_hocphan and not wants_graduation and "đợt" in q:
        wants_hocphan = any(
            x in q for x in ["học phần", "lớp học phần", "đăng ký học phần", "portal", "mở:", "đóng:"]
        )
    if not wants_hocphan and "bổ sung" in q and not wants_graduation:
        wants_hocphan = any(x in q for x in ["học phần", "lớp", "đăng ký"]) and "hồ sơ" not in q
    if wants_graduation:
        wants_hocphan = wants_hocphan and any(
            x in q
            for x in ["học phần", "đăng ký học phần", "lớp học phần", "bị hủy"]
        )
    wants_nckh = any(
        x in q
        for x in ["đề tài", "nckh", "nghiên cứu khoa học", "đăng ký đề tài", "thời hạn đăng ký"]
    )
    wants_tuyensinh = any(
        x in q
        for x in [
            "tuyển sinh",
            "xét tuyển",
            "nhập học",
            "tốt nghiệp",
            "cấp bằng",
            "ielts",
            "toefl",
            "chứng chỉ ngoại ngữ",
            "ngoại ngữ",
            "bậc 3",
            "bậc 4",
            "hồ sơ xét",
        ]
    )
    wants_list = any(
        x in q
        for x in [
            "gồm những",
            "liệt kê",
            "danh sách",
            "từng đợt",
            "những ai",
            "mốc thời gian",
            "giảng viên",
            "hướng dẫn",
        ]
    )
    wants_yes_no = any(x in q for x in ["có được", "có thể", "không", "được không"])
    return {
        "hocphan": wants_hocphan,
        "nckh": wants_nckh,
        "tuyensinh": wants_tuyensinh,
        "graduation": wants_graduation,
        "list": wants_list,
        "yes_no": wants_yes_no,
    }


def _corpus_max_year() -> int:
    if _chunks_cache:
        return infer_corpus_max_year(d.page_content for d in _chunks_cache)
    return infer_corpus_max_year()


def _is_unsupported_future_query(query: str) -> bool:
    return _guard_is_unsupported_future_query(query, max_corpus_year=_corpus_max_year())


def _query_in_corpus_domain(query: str) -> bool:
    return _guard_query_in_corpus_domain(query)


def _is_arithmetic_or_trivia_query(query: str) -> bool:
    return _guard_is_arithmetic_or_trivia_query(query)


def _is_offtopic_query(query: str) -> bool:
    return _guard_is_offtopic_query(query)


def _offtopic_refusal_response(question: str, fast_mode: bool = False) -> Dict[str, Any]:
    mode, _ = _resolve_response_mode(question, fast_mode)
    return {"answer": NO_INFO_ANSWER, "sources": [], "mode": mode}


def _doc_matches_faculty(doc: Document, faculty: str) -> bool:
    text = (doc.page_content or "").lower()
    haystack = text
    if faculty == "cntt":
        return "công nghệ thông tin" in haystack or "cntt" in haystack
    if faculty == "kinh tế":
        return "kinh tế" in haystack
    if faculty == "kinh_te_quan_tri":
        return "quản trị" in haystack or "kinh tế" in haystack
    if faculty == "kế toán":
        return "kế toán" in haystack or "tài chính" in haystack
    if faculty == "quản trị":
        return "quản trị" in haystack
    return False


def _doc_mentions_known_faculty(doc: Document) -> bool:
    text = (doc.page_content or "").lower()
    source = (doc.metadata.get("source", "") or "").lower()
    haystack = f"{text} {source}"
    known = ["công nghệ thông tin", "cntt", "khoa kinh tế", "khoa kế toán", "khoa quản trị"]
    return any(item in haystack for item in known)


def _build_gemini_payload(context: str, question: str, max_output_tokens: int) -> Dict[str, Any]:
    messages = build_prompt().format_messages(context=context, question=question)
    system_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") == "system")
    user_text = "\n\n".join(m.content for m in messages if getattr(m, "type", "") != "system")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload


def _extract_gemini_text(data: Dict[str, Any]) -> str:
    parts = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "".join(parts)


def _generate_with_gemini(context: str, question: str, max_output_tokens: int) -> str:
    payload = _build_gemini_payload(context, question, max_output_tokens)
    return _extract_gemini_text(_post_gemini(payload)).strip()


async def _stream_with_gemini(context: str, question: str, max_output_tokens: int) -> AsyncGenerator[str, None]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
    payload = _build_gemini_payload(context, question, max_output_tokens)
    timeout = httpx.Timeout(LLM_TIMEOUT, read=LLM_TIMEOUT)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, params={"alt": "sse", "key": GEMINI_API_KEY}, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    text = _extract_gemini_text(json.loads(raw))
                except json.JSONDecodeError:
                    continue
                if text:
                    yield text


def _post_gemini(payload: Dict[str, Any]) -> Dict[str, Any]:
    import time

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    last_error = None
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        for attempt in range(5):
            try:
                response = client.post(url, params={"key": GEMINI_API_KEY}, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code in (429, 503) and attempt < 4:
                    time.sleep(min(30, 3 * (2 ** attempt)))
                    continue
                raise
    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without response")


def _invoke_gemini_text(prompt: str, max_output_tokens: int = 128) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    return _extract_gemini_text(_post_gemini(payload)).strip()


def _heuristic_rewrite_query(question: str) -> str:
    """Lightweight query expansion without extra LLM calls (saves API quota)."""
    q = (question or "").strip()
    if not q:
        return q

    intent = _query_intent(q)
    rewritten = rewrite_query_for_retrieval(q, intent)

    # Học phần / đợt đăng ký: giữ CNTT nguyên để tránh kéo nhầm thông báo NCKH Khoa CNTT.
    if intent["hocphan"] and not intent["nckh"]:
        return q

    out = rewritten
    ordered = [
        ("NCKH", "nghiên cứu khoa học sinh viên"),
        ("Khoa CNTT", "Khoa Công nghệ Thông tin"),
        ("GVHD", "giảng viên hướng dẫn"),
    ]
    for src, dst in ordered:
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    if intent["nckh"] and "công nghệ thông tin" not in out.lower():
        out = re.sub(r"\bCNTT\b", "Công nghệ Thông tin", out, flags=re.IGNORECASE)
    return out


def rewrite_query(question: str, chat_history: Optional[List[dict]] = None) -> str:
    """
    Rewrite the user's question into a better, more precise query for retrieval.
    Uses deterministic heuristics to avoid query drift from small local LLMs.
    """
    if not USE_QUERY_REWRITING:
        return question
    rewritten = _heuristic_rewrite_query(question)
    if rewritten != question:
        print(f"[Query Rewriting] Heuristic: {question[:60]}... -> {rewritten[:60]}...")
    return rewritten

def init_rag(force: bool = False) -> Dict[str, Any]:
    global _initialized, _embeddings, _vectorstore, _chunks_cache, _retriever, _llm

    if force or not _initialized:
        set_index_status("indexing")
        print("[rag_service] Initializing RAG backend...")
        _embeddings = get_embeddings()
        index_path = Path(INDEX_DIR)
        index_file = index_path / "index.faiss"
        corpus_fingerprint = compute_corpus_fingerprint(DATA_DIR)
        rebuild = force or index_needs_rebuild(index_path, EMBEDDING_MODEL, corpus_fingerprint=corpus_fingerprint)

        if rebuild:
            cache_path = index_path / "chunks_cache.json"
            if index_file.exists() and not force:
                print(f"[rag_service] Rebuilding index for embedding model: {EMBEDDING_MODEL}")

            # Reparse on every rebuild. Reusing a cache after a source or chunking
            # change can produce an index that no longer matches the documents.
            raw_docs = load_pdfs(DATA_DIR)
            _chunks_cache = chunk_documents(raw_docs, _embeddings)
            to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in _chunks_cache]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False)

            _vectorstore = create_faiss_index(
                _chunks_cache,
                _embeddings,
                index_path,
                corpus_fingerprint=corpus_fingerprint,
            )
        else:
            _vectorstore = load_faiss_index(_embeddings, index_path)
            cache_path = index_path / "chunks_cache.json"
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                from langchain_core.documents import Document
                _chunks_cache = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in cached]

        _retriever = get_retriever(
            _vectorstore,
            k=INITIAL_RETRIEVE_K,
            use_hybrid=USE_HYBRID,
            chunks_for_bm25=_chunks_cache or [],
        )

        llm_options = {
            "num_predict": MAX_OUTPUT_TOKENS,
            "num_ctx": NUM_CTX,
            "num_gpu": OLLAMA_NUM_GPU,
            "num_thread": OLLAMA_NUM_THREAD,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        if _use_gemini():
            _llm = None

        else:
            _llm = OllamaLLM(model=MODEL, timeout=LLM_TIMEOUT, options=llm_options)
        print(f"[rag_service] LLM backend selected: {_llm_backend_label()}")

        _initialized = True
        set_index_status("ready")
        print("[rag_service] RAG ready.")

    return {
        "status": "ready",
        "provider": LLM_PROVIDER,
        "model": GEMINI_MODEL if _use_gemini() else MODEL,
        "hybrid": USE_HYBRID,
        "data_dir": DATA_DIR,
    }

def _fold_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", (value or "").casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace("\u0111", "d")


def _year_query_phrase_evidence(query: str, docs: List[Document]) -> Dict[str, Any]:
    """Measure whether an explicit-year query has matching content phrases in one chunk."""
    folded = _fold_evidence_text(query)
    tokens = re.findall(r"[a-z0-9]+", folded)
    years = [token for token in tokens if re.fullmatch(r"20\d{2}", token)]
    if not years:
        return {"applies": False}

    stop = {
        "theo", "tai", "lieu", "vui", "long", "tra", "cuu", "giup", "cho", "minh", "biet",
        "dua", "tren", "cac", "thong", "bao", "hien", "co", "hay", "tra", "loi", "la", "gi",
        "cua", "va", "ve", "nhung", "nao", "duoc", "khong", "mot", "nhat", "vhu",
    }
    content_tokens = [token for token in tokens if token not in stop and (len(token) > 1 or token.isdigit())]
    phrases = {
        " ".join(content_tokens[index:index + 2])
        for index in range(len(content_tokens) - 1)
        if len(content_tokens[index:index + 2]) == 2
    }
    if not phrases:
        return {"applies": True, "phrase_count": 0, "best_phrase_hits": 0, "anchor_phrase_present": False}

    anchor_phrase = " ".join(content_tokens[:2])
    best_hits = 0
    best_chunk = None
    best_anchor_phrase_hits = 0
    best_year_hits = 0
    for doc in docs:
        text = _fold_evidence_text(str(getattr(doc, "page_content", "")))
        doc_years = set(re.findall(r"20\d{2}", text))
        best_year_hits = max(best_year_hits, sum(year in doc_years for year in set(years)))
        hits = sum(phrase in text for phrase in phrases)
        if anchor_phrase in text:
            best_anchor_phrase_hits = max(best_anchor_phrase_hits, hits)
        if hits > best_hits:
            best_hits = hits
            best_chunk = {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
    return {
        "applies": True,
        "phrase_count": len(phrases),
        "best_phrase_hits": best_hits,
        "anchor_phrase": anchor_phrase,
        "best_anchor_phrase_hits": best_anchor_phrase_hits,
        "year_count": len(set(years)),
        "best_year_hits": best_year_hits,
        "best_chunk": best_chunk,
    }


def compute_evidence_confidence(docs: List[Document], query: str = ""):
    """Returns (confidence: float 0-1, top_score: float, details: dict)
    Smarter scoring that allows strong single-chunk evidence.
    """
    if not docs:
        return 0.0, 0.0, {"reason": "no docs"}

    if _embeddings is None:
        total_len = sum(len(getattr(d, 'page_content', '')) for d in docs)
        qty = min(1.0, len(docs) / 3.0)
        len_score = min(1.0, total_len / 600.0)
        conf = max(0.0, min(1.0, (qty + len_score) / 2))
        return conf, 0.3, {"top_score": 0.3, "num_docs": len(docs)}

    try:
        scored = rerank_documents(query or "", docs, _embeddings, top_k=len(docs), return_scores=True)
        scores = [float(s) for s, _ in scored]
        if not scores:
            return 0.0, 0.0, {"reason": "no scores"}

        avg = sum(scores) / len(scores)
        top_score = max(scores)
        strong_count = sum(1 for s in scores if s >= 0.22)
        strong_ratio = strong_count / len(scores)
        meaningful = sum(1 for d in docs if len(getattr(d, 'page_content', '')) > 100) / len(docs)

        conf = (avg * 0.35) + (strong_ratio * 0.35) + (meaningful * 0.3)
        conf = max(0.0, min(1.0, conf))

        return conf, top_score, {
            "num_docs": len(docs),
            "avg_score": round(avg, 4),
            "top_score": round(top_score, 4),
            "strong_ratio": round(strong_ratio, 3)
        }
    except Exception as e:
        return 0.3, 0.25, {"error": str(e)}


def _should_refuse(docs: List[Document], query: str, conf: float, top_score: float, details: dict) -> tuple[bool, str, dict]:
    """Smart refusal decision.
    - Protects strong single-chunk evidence (e.g. exact requirement like IELTS 4.0 or direct match).
    - Only refuses when evidence is genuinely weak or off-topic.
    Returns: (refuse: bool, reason: str, log_details: dict)
    """
    if not docs:
        return True, "no relevant chunks", {"num": 0, "top_score": 0.0}

    num = len(docs)
    log = {"num_chunks": num, "top_score": round(top_score, 4), "conf": round(conf, 3)}

    intent = _query_intent(query)

    # Faculty mismatch: chỉ áp dụng cho câu hỏi NCKH thuần (không lẫn học phần/đợt đăng ký).
    query_faculty = _detect_query_faculty(query)
    if query_faculty and intent["nckh"] and not intent["hocphan"] and docs:
        top_doc = docs[0]
        if _doc_mentions_known_faculty(top_doc) and not _doc_matches_faculty(top_doc, query_faculty):
            return True, f"faculty mismatch: query targets {query_faculty} but top source is different", log

    # Allow 1 very strong chunk if it has high score or key matching data
    if num == 1 and top_score >= 0.30:
        content_lower = (docs[0].page_content or "").lower()
        q_lower = (query or "").lower()
        has_specific_data = any(x in content_lower for x in ["4.0", "5.0", "6.0", "ielts", "tối thiểu", "yêu cầu", "đạt", "điểm"])
        topic_overlap = any(len(w) > 3 and w in content_lower for w in q_lower.split())
        if top_score >= 0.35 or has_specific_data or topic_overlap:
            return False, "strong single chunk with key evidence", log

    # Weak overall evidence
    if conf < 0.20:
        return True, f"low confidence ({conf:.2f})", log

    if num < 2 and top_score < 0.22:
        return True, f"weak single chunk (score {top_score:.2f})", log

    if top_score < 0.12:
        return True, f"top chunk too weak (score {top_score:.2f})", log

    return False, "sufficient evidence", log


def _inject_mandatory_chunks(query: str, intent: Dict[str, bool], docs: List[Document]) -> List[Document]:
    """Pin high-value chunks before relevance guard can drop them."""
    if not _chunks_cache:
        return docs

    mandatory, _ = find_mandatory_chunks(_chunks_cache, query, intent)
    if not mandatory:
        return docs

    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id"), d.page_content[:80]) for d in docs}
    merged = []
    for extra in mandatory:
        key = (extra.metadata.get("source"), extra.metadata.get("chunk_id"), extra.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(extra)
    return (merged + list(docs))[:MAX_CONTEXT_CHUNKS]


def _supplement_from_chunk_cache(query: str, docs: List[Document], max_add: int = 3) -> List[Document]:
    """Pull high-value table/list chunks from the indexed cache when retrieval under-fetches."""
    if not _chunks_cache:
        return docs

    q = (query or "").lower()
    intent = _query_intent(query)
    scored_extras: List[tuple[int, Document]] = []

    for chunk in _chunks_cache:
        score = score_supplement_chunk(query, intent, chunk)
        if score > 0:
            scored_extras.append((score, chunk))

    if not scored_extras:
        return docs

    lecturer_query = ("gi???ng vi??n" in q or "h?????ng d???n" in q) and (
        intent["nckh"] or "????? t??i" in q or "nghi??n c???u" in q
    )
    if lecturer_query:
        return docs
    extra_limit = max_add

    scored_extras.sort(key=lambda item: item[0], reverse=True)
    seen = {(d.metadata.get("source"), d.metadata.get("chunk_id"), d.page_content[:80]) for d in docs}
    merged = list(docs)
    for _, extra in scored_extras[:extra_limit]:
        key = (extra.metadata.get("source"), extra.metadata.get("chunk_id"), extra.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.insert(0, extra)
    return merged[:MAX_CONTEXT_CHUNKS]


def _docs_to_sources(docs: List[Document], limit: int = 3) -> List[Dict]:
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}|{page}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page, "text": strip_doc_display_prefix(doc.page_content)})
        if len(sources) >= limit:
            break
    return sources


def _get_retriever_for_request(
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
):
    """Build a per-request retriever when hybrid or source filter overrides are set."""
    if use_hybrid is None and not filter_source:
        return _retriever

    if not _initialized:
        init_rag()

    effective_hybrid = USE_HYBRID if use_hybrid is None else use_hybrid
    metadata_filter: Dict[str, Any] = {}
    if filter_source:
        metadata_filter["source_contains"] = filter_source

    return get_retriever(
        _vectorstore,
        k=INITIAL_RETRIEVE_K,
        use_hybrid=effective_hybrid,
        chunks_for_bm25=_chunks_cache or [],
        metadata_filter=metadata_filter or None,
    )


def _retrieve_pure_docs(
    question: str,
    chat_history: Optional[List[dict]] = None,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> tuple[str, List[Document], bool, str, Dict[str, Any]]:
    """Retriever-only path. Do not add answer-specific routing or document pinning here."""
    if not _initialized:
        init_rag()
    if _is_offtopic_query(question):
        return question, [], True, "off-topic query", {"num_chunks": 0, "top_score": 0.0, "conf": 0.0}
    if _is_unsupported_future_query(question):
        return question, [], True, "future year outside corpus", {"num_chunks": 0, "top_score": 0.0, "conf": 0.0}

    pipeline_started = time.perf_counter()
    # Original wording remains the ranking/guard query. Local expansion adds
    # alternate lexical forms only, so exact terms still drive the ranking.
    query_expansion = normalize_query_for_retrieval(question)
    rewritten = rewrite_query(question, chat_history) if USE_QUERY_REWRITING else question
    normalized_query = str(query_expansion["normalized_query"])
    if query_expansion["used"] and query_expansion["changed"]:
        rewritten = normalized_query
        normalized_rewritten = normalized_query
    else:
        normalized_rewritten = (
            rewrite_query(normalized_query, chat_history)
            if USE_QUERY_REWRITING
            else normalized_query
        )
    query_rewrite_seconds = time.perf_counter() - pipeline_started
    retriever = _get_retriever_for_request(use_hybrid, filter_source)
    retrieval_started = time.perf_counter()
    retrieval_query = normalized_rewritten if query_expansion["used"] and query_expansion["changed"] else rewritten
    candidates = retriever.invoke(retrieval_query) if retriever else []
    if not candidates and retrieval_query != rewritten and retriever:
        candidates = retriever.invoke(rewritten)
    # Preserve strong lexical evidence for codes, cohorts, dates and unaccented queries.
    # Keep semantic/hybrid leaders, then interleave a small BM25 reserve before reranking.
    lexical_candidates = []
    bm25_ret = getattr(retriever, "bm25_ret", None)
    if bm25_ret is not None:
        try:
            lexical_candidates = bm25_ret.search(retrieval_query, min(5, INITIAL_RETRIEVE_K))
        except Exception:
            lexical_candidates = []
    if lexical_candidates:
        balanced = list(candidates[:5]) + list(lexical_candidates) + list(candidates[5:])
        seen_keys = set()
        candidates = []
        for doc in balanced:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("chunk_id"), doc.page_content[:80])
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(doc)
    vector_retrieval_seconds = time.perf_counter() - retrieval_started
    keyword_retrieval_seconds = 0.0
    candidates = candidates[:max(INITIAL_RETRIEVE_K, 10)]
    rerank_started = time.perf_counter()
    if USE_RERANKER and _embeddings is not None and candidates:
        scored = rerank_documents(rewritten, candidates, _embeddings, top_k=FINAL_TOP_K, return_scores=True)
        docs = []
        for score, doc in scored:
            doc.metadata["_pure_rerank_score"] = round(float(score), 6)
            docs.append(doc)
    else:
        docs = candidates[:FINAL_TOP_K]

    if USE_RELEVANCE_GUARD and _embeddings is not None and docs:
        docs = filter_relevant_chunks(
            rewritten, docs, _embeddings,
            min_score=MIN_RELEVANCE_SCORE,
            min_chunks=MIN_RELEVANT_CHUNKS,
            strict=True,
        )
    # Retain a tiny lexical evidence reserve after semantic relevance filtering.
    # This protects exact cohorts, notice codes and dates in long tabular chunks.
    seen_docs = set()
    balanced_docs = []
    for candidate_doc in list(lexical_candidates[:2]) + list(docs):
        key = (candidate_doc.metadata.get("source"), candidate_doc.metadata.get("page"), candidate_doc.metadata.get("chunk_id"))
        if key not in seen_docs:
            balanced_docs.append(candidate_doc)
            seen_docs.add(key)
    docs = balanced_docs
    if is_nckh_lecturer_query(question):
        docs = prefer_nckh_lecturer_chunks(docs)
    primary_chunks = [_chunk_audit_record(doc) for doc in docs]
    docs, neighbor_candidates = _expand_pure_neighbors(rewritten, docs, MAX_CONTEXT_CHUNKS)
    conf, top_score, details = compute_evidence_confidence(docs, rewritten)
    rerank_seconds = time.perf_counter() - rerank_started
    year_phrase_evidence = _year_query_phrase_evidence(question, docs)
    logd = {
        "year_phrase_evidence": year_phrase_evidence,
        "query_expansion": {**query_expansion, "normalized_rewritten_query": normalized_rewritten},
        "query_rewrite_seconds": round(query_rewrite_seconds, 4),
        "keyword_retrieval_seconds": round(keyword_retrieval_seconds, 4),
        "vector_retrieval_seconds": round(vector_retrieval_seconds, 4),
        "rerank_seconds": round(rerank_seconds, 4),
        "num_chunks": len(docs), "top_score": top_score, "conf": conf, **details,
        "primary_chunks": primary_chunks,
        "neighbor_candidates": neighbor_candidates,
        "selected_context_chunks": [_chunk_audit_record(doc) for doc in docs],
    }
    if year_phrase_evidence.get("applies") and (
        year_phrase_evidence.get("best_year_hits", 0) < year_phrase_evidence.get("year_count", 0)
        or year_phrase_evidence.get("best_anchor_phrase_hits", 0) == 0
    ):
        return rewritten, docs, True, "requested year lacks matching evidence", logd
    if not docs or conf < float(os.getenv("PURE_RAG_MIN_CONFIDENCE", "0.30")):
        return rewritten, docs, True, "insufficient relevant context", logd
    return rewritten, docs, False, "", logd

def _retrieve_filtered_docs(
    question: str,
    chat_history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> tuple[str, List[Document], bool, str, Dict[str, Any]]:
    """
    Pure RAG pipeline:
    rewrite (optional) -> retriever/FAISS -> rerank -> relevance guard -> refusal check.
    """
    if PURE_RAG_MODE:
        return _retrieve_pure_docs(question, chat_history, use_hybrid, filter_source)

    if not _initialized:
        init_rag()

    if _is_offtopic_query(question):
        return question, [], True, "off-topic query", {"num_chunks": 0, "top_score": 0.0, "conf": 0.0}

    if _is_unsupported_future_query(question):
        return question, [], True, "future year outside corpus", {"num_chunks": 0, "top_score": 0.0, "conf": 0.0}

    intent = _query_intent(question)
    effective_final_k = 4 if fast_mode else FINAL_TOP_K
    if intent["list"]:
        effective_final_k = min(MAX_CONTEXT_CHUNKS, effective_final_k + 2)
    # Fast mode skips optional rewrite and reranking to prioritize response time.
    do_rewrite = USE_QUERY_REWRITING and not fast_mode
    do_rerank = USE_RERANKER and not fast_mode
    min_chunks = 1 if intent["list"] or intent["yes_no"] or intent["hocphan"] else MIN_RELEVANT_CHUNKS

    retrieval_query = rewrite_query(question, chat_history) if do_rewrite else question
    retriever = _get_retriever_for_request(use_hybrid, filter_source)
    raw_docs = retriever.invoke(retrieval_query) if retriever else []
    docs = light_keyword_boost_reorder(retrieval_query, raw_docs, top_k=len(raw_docs))

    hybrid_top = raw_docs[:6]
    rerank_k = min(len(docs), max(effective_final_k, 6 if fast_mode else 8))
    if do_rerank and _embeddings is not None and len(docs) > effective_final_k:
        docs = rerank_documents(retrieval_query, docs, _embeddings, top_k=rerank_k)
    else:
        docs = docs[:rerank_k]

    seen = set()
    merged = []
    for doc in hybrid_top + docs:
        key = (doc.metadata.get("source"), doc.metadata.get("chunk_id"), doc.page_content[:60])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    docs = merged[:rerank_k]
    docs = expand_adjacent_chunks(docs, _chunks_cache, window=NEIGHBOR_CHUNK_WINDOW, max_chunks=MAX_CONTEXT_CHUNKS)
    docs = _inject_mandatory_chunks(question, intent, docs)
    docs = prefer_primary_source(retrieval_query, docs, min_chunks=1 if intent["hocphan"] else 2)[:MAX_CONTEXT_CHUNKS]

    if USE_RELEVANCE_GUARD and _embeddings is not None and docs:
        guard_min_score = MIN_RELEVANCE_SCORE * (0.85 if fast_mode else 1.0)
        docs = filter_relevant_chunks(
            retrieval_query,
            docs,
            _embeddings,
            min_score=guard_min_score,
            min_chunks=min_chunks,
            strict=not fast_mode,
        )

    pinned_docs, pin_only_name = (
        find_mandatory_chunks(_chunks_cache or [], question, intent)
        if _chunks_cache
        else ([], None)
    )

    # Inject table/list chunks after guard so high-value schedule rows are not dropped.
    docs = _supplement_from_chunk_cache(retrieval_query, docs)[:MAX_CONTEXT_CHUNKS]

    if pinned_docs:
        seen = {(d.metadata.get("source"), d.metadata.get("chunk_id"), d.page_content[:80]) for d in docs}
        for extra in pinned_docs:
            key = (extra.metadata.get("source"), extra.metadata.get("chunk_id"), extra.page_content[:80])
            if key not in seen:
                docs.insert(0, extra)
                seen.add(key)
        docs = docs[:MAX_CONTEXT_CHUNKS]

    if ("giảng viên" in (question or "").lower() or "hướng dẫn" in (question or "").lower()) and intent["nckh"]:
        docs = prefer_nckh_lecturer_chunks(docs)

    q_lower = (question or "").lower()
    ielts_query = any(x in q_lower for x in ["ielts", "toefl", "chứng chỉ ngoại ngữ"])
    bosung_query = intent["hocphan"] and any(x in q_lower for x in ["bổ sung", "bị hủy", "hủy"])

    if ielts_query:
        docs = prefer_ielts_cert_chunks(docs)

    if intent.get("tuyensinh") and not intent["nckh"] and not intent["hocphan"] and not intent.get("graduation") and not ielts_query:
        docs = prefer_tuyensinh_narrative_chunks(docs)
        if fast_mode:
            docs = docs[:3]

    _PIN_ONLY_LIMITS = {
        "ielts": 2,
        "hocphan_bosung": 1,
        "hp_k2024_dot5": 1,
        "song_nganh_credits": 1,
        "graduation_hoso": 2,
        "cap_bang": 1,
        "thi_hk": 1,
        "song_nganh_fee": 1,
    }
    if pin_only_name in _PIN_ONLY_LIMITS and pinned_docs:
        docs = pinned_docs[: _PIN_ONLY_LIMITS[pin_only_name]]
    elif pin_only_name == "graduation" and pinned_docs:
        docs = (pinned_docs[:2] + [d for d in docs if d not in pinned_docs])[:MAX_CONTEXT_CHUNKS]

    conf, top_score, details = compute_evidence_confidence(docs, retrieval_query)
    refuse, reason, logd = _should_refuse(docs, retrieval_query, conf, top_score, details)
    if refuse:
        return retrieval_query, docs, True, reason, logd

    query_faculty = _detect_query_faculty(retrieval_query)
    apply_faculty_guard = query_faculty and intent["nckh"] and not intent["hocphan"]
    if apply_faculty_guard and docs:
        top_doc = docs[0]
        if _doc_mentions_known_faculty(top_doc) and not _doc_matches_faculty(top_doc, query_faculty):
            return retrieval_query, docs, True, f"faculty mismatch (expected {query_faculty})", logd

    return retrieval_query, docs, False, "", logd


def _log_refusal(retrieval_query: str, docs: List[Document], reason: str, logd: Dict[str, Any]):
    top_src = docs[0].metadata.get("source", "?") if docs else "?"
    top_sc = logd.get("top_score", 0.0)
    print("[Refusal] Returning no-context response")
    print(f"  Confidence: {logd.get('conf', 0):.3f} | Chunks: {logd.get('num_chunks', len(docs))} | Top source: {top_src} | Top score: {top_sc:.3f}")
    print(f"  Reason: {reason}")


def _maybe_extract_nckh_facts(
    question: str,
    docs: List[Document],
) -> Optional[tuple[str, List[Dict[str, str]]]]:
    """Structured NCKH facts parsed from PDF chunks."""
    if "tình huống" in (question or "").lower():
        return None
    return try_extract_nckh_facts(question, docs, _chunks_cache)


def _maybe_extract_policy_facts(
    question: str,
    docs: List[Document],
) -> Optional[tuple[str, List[Dict[str, str]]]]:
    """Parse credits/fees/dates/IELTS bands from indexed chunks."""
    if "tình huống" in (question or "").lower():
        return None
    return try_extract_policy_facts(question, docs, _chunks_cache)


def _try_pre_retrieval_extract(
    question: str,
) -> Optional[tuple[str, List[Dict[str, str]]]]:
    """Fast path: answer from chunks_cache before embedding retrieval."""
    extracted = _maybe_extract_nckh_facts(question, [])
    if extracted:
        return extracted
    return _maybe_extract_policy_facts(question, [])


def _is_refusal_answer(answer: str) -> bool:
    text = (answer or "").strip()
    return (
        not text
        or text == NO_INFO_ANSWER
        or text.startswith("Không tìm thấy")
    )


def _retry_focused_answer(question: str, context: str, model: str) -> str:
    """Second pass with a shorter prompt for yes/no and comparison questions."""
    intent = _query_intent(question)
    q_lower = (question or "").lower()

    if intent["yes_no"]:
        instruction = (
            "Trả lời bắt đầu bằng 'Có.' hoặc 'Không.' rồi giải thích ngắn CHỈ từ CONTEXT. "
            "Nếu CONTEXT nêu điều kiện đủ (ví dụ từ năm thứ 2 trở lên), suy luận rõ cho năm 1."
        )
    elif any(x in q_lower for x in ["giống nhau", "khác nhau", "so sánh"]):
        instruction = (
            "So sánh các mốc thời gian trong CONTEXT. "
            "Trả lời có giống nhau không và nêu rõ từng thời hạn."
        )
    elif intent.get("tuyensinh") and not intent["nckh"] and not intent["hocphan"]:
        instruction = (
            "CONTEXT chứa thông báo tuyển sinh của Trường Đại học Văn Hiến. "
            "Tóm tắt ngắn gọn nội dung chính: trình độ/ngành, đối tượng, hình thức tuyển sinh, "
            "thời hạn hoặc liên hệ nếu có. Chỉ dùng CONTEXT. "
            "Nếu CONTEXT có thông tin tuyển sinh thì PHẢI trả lời, không từ chối."
        )
    else:
        return ""

    prompt = f"""{instruction}

CONTEXT:
{context[:3800]}

Câu hỏi: {question}

Trả lời:"""

    llm = OllamaLLM(
        model=model,
        timeout=LLM_TIMEOUT,
        options={"num_predict": 256, "num_ctx": NUM_CTX, "temperature": 0.0},
    )
    return llm.invoke(prompt).strip()


def _output_token_budget(fast_mode: bool, num_questions: int = 1) -> int:
    base = FAST_MODE_MAX_OUTPUT_TOKENS if fast_mode else MAX_OUTPUT_TOKENS
    if num_questions <= 1:
        return base
    return min(MAX_OUTPUT_TOKENS, base + (num_questions - 1) * 180)


def _generate_answer_from_docs(
    question: str,
    docs: List[Document],
    fast_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> str:
    context = format_context(docs, question)
    if max_tokens is None:
        max_tokens = _output_token_budget(fast_mode)
    fallback_model = OLLAMA_FALLBACK_MODEL if LLM_PROVIDER == "gemini" else MODEL

    if _use_gemini():
        try:
            _log_llm_backend("sync", question)
            answer = _generate_with_gemini(context, question, max_tokens)
            if not _is_refusal_answer(answer):
                return answer
            print("[LLM] Gemini returned refusal, trying focused Ollama retry")
        except Exception as e:
            print(f"[Gemini] Generate failed, trying Ollama fallback ({OLLAMA_FALLBACK_MODEL}): {e}")

    _log_llm_backend("sync-fallback" if LLM_PROVIDER == "gemini" else "sync", question)
    prompt = build_prompt()
    intent = _query_intent(question)
    tuyensinh_fast = fast_mode and intent.get("tuyensinh") and not intent["nckh"] and not intent["hocphan"]
    ollama_opts = {
        "num_predict": max_tokens,
        "num_ctx": 2048 if tuyensinh_fast else (1024 if fast_mode else NUM_CTX),
        "num_gpu": OLLAMA_NUM_GPU,
        "num_thread": OLLAMA_NUM_THREAD,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if fast_mode:
        llm = OllamaLLM(model=fallback_model, timeout=LLM_TIMEOUT, options=ollama_opts)
    else:
        llm = _llm if LLM_PROVIDER != "gemini" and _llm else OllamaLLM(
            model=fallback_model,
            timeout=LLM_TIMEOUT,
            options=ollama_opts,
        )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    if not _is_refusal_answer(answer):
        return answer

    retry = _retry_focused_answer(question, context, fallback_model)
    return retry or answer


class PureGenerationTimeout(Exception):
    pass


def _is_list_request(question: str) -> bool:
    return bool(re.search(r"\b(list|enumerate|liệt kê|danh sách|tất cả)\b", question, re.IGNORECASE))


def _numbered_rows_in_chunk(text: str) -> int:
    return len(re.findall(r"(?:^|\n|\|)\s*\d{1,3}\s+[A-Za-zÀ-ỹ]", text))


def _parse_json_records(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


async def _extract_list_records_by_chunk(client: httpx.AsyncClient, question: str, docs: List[Document], timeout_seconds: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for doc in docs:
        expected = _numbered_rows_in_chunk(doc.page_content)
        prompt = (
            "Extract every numbered record from this DOCUMENT CHUNK as strict JSON only: "
            "[{\"name\": string, \"email\": string|null, \"details\": string|null}]. "
            "Do not summarize and do not omit records.\n\nDOCUMENT CHUNK:\n" + doc.page_content
        )
        extracted: List[Dict[str, Any]] = []
        attempts = 0
        while attempts < 2:
            attempts += 1
            response = await client.post("http://127.0.0.1:11434/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.0, "num_predict": MAX_OUTPUT_TOKENS, "num_ctx": 2048, "keep_alive": OLLAMA_KEEP_ALIVE}})
            response.raise_for_status()
            extracted = _parse_json_records(response.json().get("response", ""))
            if len(extracted) >= expected:
                break
        for record in extracted:
            record["source"] = doc.metadata.get("source")
            record["page"] = doc.metadata.get("page")
            record["chunk_id"] = doc.metadata.get("chunk_id")
            records.append(record)
        audit.append({"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "chunk_id": doc.metadata.get("chunk_id"), "expected_numbered_rows": expected, "extracted_records": len(extracted), "attempts": attempts})
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = re.sub(r"\s+", " ", str(record.get("name", "")).strip().casefold())
        if key and key not in seen:
            seen.add(key)
            deduped.append(record)
    return deduped, audit


def _format_list_answer(question: str, records: List[Dict[str, Any]]) -> str:
    wants_email = bool(re.search(r"\b(email|e-mail|liên hệ)\b", question, re.IGNORECASE))
    lines = []
    for index, record in enumerate(records, 1):
        name = str(record.get("name", "")).strip()
        if not name:
            continue
        line = f"{index}. {name}"
        if wants_email and record.get("email"):
            line += f" - {record['email']}"
        line += f" [{record.get('source')}, trang {record.get('page')}]"
        lines.append(line)
    return "\n".join(lines) if lines else NO_INFO_ANSWER

async def ask_question_async(question: str, use_hybrid: Optional[bool] = None, filter_source: Optional[str] = None, history: Optional[List[dict]] = None, fast_mode: bool = False) -> Dict[str, Any]:
    """Async PURE_RAG generation with a cancellable HTTP read timeout."""
    if not PURE_RAG_MODE:
        return ask_question(question, use_hybrid, filter_source, history, fast_mode)
    request_started = asyncio.get_running_loop().time()
    generation_timeout = float(os.getenv("PURE_RAG_GENERATION_TIMEOUT", "180"))
    retrieval_query, docs, refused, reason, logd = _retrieve_filtered_docs(question, history, fast_mode, use_hybrid, filter_source)
    retrieval_ms = round((asyncio.get_running_loop().time() - request_started) * 1000, 1)
    if refused:
        structured = _maybe_extract_nckh_facts(question, docs) or _maybe_extract_policy_facts(question, docs)
        if structured:
            answer, sources = structured
            _write_rag_audit(question, retrieval_query, docs, {
                **logd, "structured_refusal_fallback_used": True,
                "generated_answer": answer, "technical_status": "completed",
            }, False, "")
            return {"answer": answer, "sources": sources, "mode": "pure_rag", "technical_status": None}
        _write_rag_audit(question, retrieval_query, docs, logd, True, reason)
        return {"answer": NO_INFO_ANSWER, "sources": [], "mode": "pure_rag", "technical_status": None}
    # Reuse deterministic structured extractors before generic answer-shape
    # handling. They parse the indexed PDF evidence and return cited answers.
    lecturer_fact = try_extract_nckh_lecturers(question, docs, _chunks_cache)
    if lecturer_fact:
        answer, sources = lecturer_fact
        _write_rag_audit(question, retrieval_query, docs, {
            **logd, "lecturer_grounding_used": True, "generated_answer": answer,
            "technical_status": "completed",
        }, False, "")
        return {"answer": answer, "sources": sources, "mode": "pure_rag", "technical_status": None}
    nckh_fact = _maybe_extract_nckh_facts(question, docs)
    if nckh_fact:
        answer, sources = nckh_fact
        _write_rag_audit(question, retrieval_query, docs, {
            **logd, "nckh_fact_grounding_used": True, "generated_answer": answer,
            "technical_status": "completed",
        }, False, "")
        return {"answer": answer, "sources": sources, "mode": "pure_rag", "technical_status": None}

    policy_fact = _maybe_extract_policy_facts(question, docs)
    if policy_fact:
        answer, sources = policy_fact
        _write_rag_audit(question, retrieval_query, docs, {
            **logd, "policy_fact_grounding_used": True, "generated_answer": answer,
            "technical_status": "completed",
        }, False, "")
        return {"answer": answer, "sources": sources, "mode": "pure_rag", "technical_status": None}

    # The evidence selector ranks the complete post-guard context. Restricting it
    # to the first chunk can discard a directly adjacent temporal fact.
    answer_shape = classify_answer_shape(question, docs)
    if answer_shape["answer_shape"] == "boolean":
        boolean_fact = extract_boolean_fact(question, docs)
        if boolean_fact is not None:
            evidence_line, evidence_doc = boolean_fact
            source = str(evidence_doc.metadata.get("source", ""))
            page = int(evidence_doc.metadata.get("page", 0)) + 1
            chunk_id = evidence_doc.metadata.get("chunk_id")
            answer = evidence_line + f"\n[Ngu?n: {source}, trang {page}, chunk {chunk_id}]"
            _write_rag_audit(question, retrieval_query, docs, {
                **logd, **answer_shape, "boolean_grounding_used": True, "evidence_quote": evidence_line,
                "generated_answer": answer, "technical_status": "completed",
            }, False, "")
            return {"answer": answer, "sources": _docs_to_sources([evidence_doc]), "mode": "pure_rag", "technical_status": None}
    location_fact = extract_location_fact(question, docs)
    if location_fact is not None:
        evidence_line, evidence_doc = location_fact
        source = str(evidence_doc.metadata.get("source", ""))
        page = int(evidence_doc.metadata.get("page", 0)) + 1
        chunk_id = evidence_doc.metadata.get("chunk_id")
        answer = evidence_line + f"\n[Nguồn: {source}, trang {page}, chunk {chunk_id}]"
        _write_rag_audit(question, retrieval_query, docs, {
            **logd, "location_grounding_used": True, "evidence_quote": evidence_line,
            "generated_answer": answer, "technical_status": "completed",
        }, False, "")
        return {"answer": answer, "sources": _docs_to_sources([evidence_doc]), "mode": "pure_rag", "technical_status": None}
    # The indexed cache is a deterministic lexical fallback for facts that live
    # in a short adjacent chunk and can be displaced by the context-size cap.
    grounding_docs = list(docs)
    seen_grounding = {(doc.metadata.get("source"), doc.metadata.get("chunk_id")) for doc in grounding_docs}
    for cached_doc in (_chunks_cache or []):
        key = (cached_doc.metadata.get("source"), cached_doc.metadata.get("chunk_id"))
        if key not in seen_grounding:
            grounding_docs.append(cached_doc)
            seen_grounding.add(key)
    extractive = extract_extractive_fact(question, grounding_docs)
    if extractive is not None:
        evidence_line, evidence_doc = extractive
        source = str(evidence_doc.metadata.get("source", ""))
        page = int(evidence_doc.metadata.get("page", 0)) + 1
        chunk_id = evidence_doc.metadata.get("chunk_id")
        answer = evidence_line + f"\n[Ngu?n: {source}, trang {page}, chunk {chunk_id}]"
        _write_rag_audit(question, retrieval_query, docs, {
            **logd, "extractive_grounding_used": True, "evidence_quote": evidence_line,
            "generated_answer": answer, "technical_status": "completed",
        }, False, "")
        return {"answer": answer, "sources": _docs_to_sources([evidence_doc]), "mode": "pure_rag", "technical_status": None}
    answer_shape = classify_answer_shape(question, docs)
    list_extraction_triggered = answer_shape["answer_shape"] == "list"
    logd = {**logd, **answer_shape, "list_extraction_triggered": list_extraction_triggered}
    if list_extraction_triggered:
        table_rows = valid_rows_for_sources(Path(INDEX_DIR) / "table_store.sqlite", [doc.metadata.get("source", "") for doc in docs])
        if table_rows:
            grouped = {}
            for row in table_rows:
                key = (row["source"], row["page"], row["chunk_id"])
                grouped.setdefault(key, []).append(row)
            sections = []
            for (source, page, chunk_id), rows in grouped.items():
                # Keep each table-page heading separate from its ordered Markdown list.
                lines = [f"**Nguồn: {source}, trang {int(page) + 1}, chunk {chunk_id}**", ""]
                for row in rows:
                    raw = " ".join(str(row["raw_text"]).split())
                    raw = re.sub(r"^\d+\s+", "", raw)
                    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw)
                    name = raw[:email.start()].strip() if email else raw
                    lines.append(f"{row['row_number']}. {name}" + (f" - {email.group(0)}" if email else ""))
                sections.append("\n".join(lines))
            answer = "\n\n".join(sections)
            _write_rag_audit(question, retrieval_query, docs, {**logd, "table_rag_used": True, "table_row_count": len(table_rows), "generated_answer": answer, "technical_status": "completed"}, False, "")
            return {"answer": answer, "sources": _docs_to_sources(docs), "mode": "pure_rag", "technical_status": None}
        # Table store has no matching valid table: continue with ordinary PURE_RAG.
        async def generate_json(prompt: str) -> str:
            timeout = httpx.Timeout(connect=10.0, read=generation_timeout, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
                          "options": {"temperature": 0.0, "num_predict": MAX_OUTPUT_TOKENS,
                                      "num_ctx": 2048, "keep_alive": OLLAMA_KEEP_ALIVE}},
                )
                response.raise_for_status()
                return response.json().get("response", "")

        try:
            async with asyncio.timeout(generation_timeout):
                records, extraction_audit = await extract_records_by_chunk(question, docs, generate_json)
        except (TimeoutError, httpx.HTTPError):
            records, extraction_audit = [], []
        if records:
            wants_email = bool(re.search(r"\b(email|e-mail|liên hệ)\b", question, re.IGNORECASE))
            lines = []
            for number, record in enumerate(records, 1):
                name = str(record.get("name", "")).strip()
                if not name:
                    continue
                line = f"{number}. {name}"
                if wants_email and record.get("email"):
                    line += f" - {record['email']}"
                lines.append(f"{line} [{record.get('source')}, trang {record.get('page')}, chunk {record.get('chunk_id')}]")
            answer = "\n".join(lines) or NO_INFO_ANSWER
            _write_rag_audit(question, retrieval_query, docs, {
                **logd, "generated_answer": answer, "list_extraction": extraction_audit,
                "extracted_records": records, "extracted_record_count": len(records),
                "technical_status": "completed",
            }, False, "")
            return {"answer": answer, "sources": _docs_to_sources(docs), "mode": "pure_rag", "technical_status": None}
    context_started = asyncio.get_running_loop().time()
    full_context = format_context(docs, question)
    context_ms = round((asyncio.get_running_loop().time() - context_started) * 1000, 1)
    options = {
        "temperature": PURE_RAG_TEMPERATURE,
        "num_predict": min(PURE_RAG_MAX_OUTPUT_TOKENS, FAST_MODE_MAX_OUTPUT_TOKENS) if fast_mode else PURE_RAG_MAX_OUTPUT_TOKENS,
        "num_ctx": min(PURE_RAG_NUM_CTX, 1024) if fast_mode else PURE_RAG_NUM_CTX,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    retry_context, retry_docs = select_evidence_window(question, docs)
    attempts = [("initial", full_context, docs)]
    if retry_context and retry_docs:
        attempts.append(("evidence_retry", retry_context, retry_docs))
    grounding_attempts = []
    generation_seconds = 0.0

    for attempt_name, attempt_context, attempt_docs in attempts:
        system_prompt, user_prompt, final_prompt = build_grounding_prompts(
            question, attempt_context, retry=attempt_name == "evidence_retry"
        )
        payload = {
            "model": MODEL,
            "prompt": final_prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "source": {"type": "string"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "integer"},
                },
                "required": ["answer", "evidence_quote", "source", "page", "chunk_id"],
            },
            "options": options,
        }
        generation_started = asyncio.get_running_loop().time()
        timeout = httpx.Timeout(
            connect=min(10.0, generation_timeout),
            read=generation_timeout,
            write=min(10.0, generation_timeout),
            pool=min(10.0, generation_timeout),
        )
        raw_output = ""
        ollama_result: Dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with asyncio.timeout(generation_timeout):
                    response = await client.post("http://127.0.0.1:11434/api/generate", json=payload)
                response.raise_for_status()
                ollama_result = response.json()
                raw_output = str(ollama_result.get("response", "")).strip()
        except (TimeoutError, httpx.ReadTimeout) as exc:
            timeout_log = {
                **logd,
                "technical_status": "generation_timeout",
                "grounding_attempts": grounding_attempts,
                "timing_ms": {
                    "retrieval": retrieval_ms,
                    "context": context_ms,
                    "generation": round((asyncio.get_running_loop().time() - generation_started) * 1000, 1),
                    "total": round((asyncio.get_running_loop().time() - request_started) * 1000, 1),
                },
            }
            _write_rag_audit(question, retrieval_query, docs, timeout_log, False, "")
            raise PureGenerationTimeout("generation_timeout") from exc
        except httpx.HTTPError as exc:
            raise PureGenerationTimeout("generation_backend_error") from exc

        attempt_seconds = asyncio.get_running_loop().time() - generation_started
        generation_seconds += attempt_seconds
        parsed = parse_grounded_json(raw_output)
        valid, validation_errors, grounded = validate_grounded_response(parsed, attempt_context, attempt_docs, question)
        attempt_debug = {
            "attempt": attempt_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "final_prompt": final_prompt,
            "context_sent_to_ollama": attempt_context,
            "context_chars": len(attempt_context),
            "ollama_options": options,
            "generated_output_raw": raw_output,
            "parsed_output": parsed,
            "validation_passed": valid,
            "validation_errors": validation_errors,
            "source_page_chunk": [
                {"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "chunk_id": doc.metadata.get("chunk_id")}
                for doc in attempt_docs
            ],
            "ollama_eval_count": ollama_result.get("eval_count"),
            "ollama_done_reason": ollama_result.get("done_reason"),
            "generation_seconds": round(attempt_seconds, 4),
        }
        grounding_attempts.append(attempt_debug)
        if not valid:
            continue

        cited_doc = next(
            doc for doc in attempt_docs
            if str(doc.metadata.get("source", "")) == grounded["source"]
            and int(doc.metadata.get("page")) + 1 == grounded["page"]
            and str(doc.metadata.get("chunk_id")) == str(grounded["chunk_id"])
        )
        answer = (
            grounded["answer"]
            + f"\n[Nguồn: {grounded['source']}, trang {grounded['page']}, chunk {grounded['chunk_id']}]"
        )
        completed_log = {
            **logd,
            "generated_answer": answer,
            "technical_status": "completed",
            "ollama_eval_count": ollama_result.get("eval_count"),
            "ollama_done_reason": ollama_result.get("done_reason"),
            "grounding_attempts": grounding_attempts,
            "generation_debug": attempt_debug,
            "timing_ms": {
                "retrieval": retrieval_ms,
                "context": context_ms,
                "generation": round(generation_seconds * 1000, 1),
                "total": round((asyncio.get_running_loop().time() - request_started) * 1000, 1),
            },
        }
        _write_rag_audit(question, retrieval_query, docs, completed_log, False, "")
        return {"answer": answer, "sources": _docs_to_sources([cited_doc]), "mode": "pure_rag", "technical_status": None}

    failed_log = {
        **logd,
        "generated_answer": None,
        "technical_status": "grounding_validation_failed",
        "grounding_attempts": grounding_attempts,
        "generation_debug": grounding_attempts[-1] if grounding_attempts else {},
        "timing_ms": {
            "retrieval": retrieval_ms,
            "context": context_ms,
            "generation": round(generation_seconds * 1000, 1),
            "total": round((asyncio.get_running_loop().time() - request_started) * 1000, 1),
        },
    }
    _write_rag_audit(question, retrieval_query, docs, failed_log, False, "grounding validation failed")
    return {"answer": NO_INFO_ANSWER, "sources": [], "mode": "pure_rag", "technical_status": "grounding_validation_failed"}

def get_response_mode_for_question(question: str, fast_mode: bool = False) -> str:
    """Return the single supported response mode for SSE metadata."""
    return "rag"


def get_retrieved_context(
    question: str,
    chat_history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> List[Dict]:
    """Retrieve context chunks via the unified pure-RAG pipeline."""
    _, docs, refused, reason, logd = _retrieve_filtered_docs(
        question, chat_history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    )
    if refused:
        _log_refusal(question, docs, reason, logd)
        return []
    return [
        {"content": d.page_content, "source": d.metadata.get("source", ""), "page": d.metadata.get("page", "?")}
        for d in docs
    ]


def get_sources_for_question(
    question: str,
    chat_history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    limit: int = 3,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> List[Dict]:
    """Return sources from retrieval only, without generating an answer."""
    sub_questions = [question] if PURE_RAG_MODE else _questions_for_pipeline(question, fast_mode)
    if len(sub_questions) > 1:
        return _merge_sources(
            [
                get_sources_for_question(
                    sq,
                    chat_history=chat_history,
                    fast_mode=fast_mode,
                    limit=limit,
                    use_hybrid=use_hybrid,
                    filter_source=filter_source,
                )
                for sq in sub_questions
            ],
            limit=limit,
        )

    _, docs, refused, _, _ = _retrieve_filtered_docs(
        question, chat_history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    )
    if not refused:
        extracted = try_extract_nckh_lecturers(question, docs, _chunks_cache)
        if extracted:
            return extracted[1][:limit]

    contexts = get_retrieved_context(
        question,
        chat_history=chat_history,
        fast_mode=fast_mode,
        use_hybrid=use_hybrid,
        filter_source=filter_source,
    )
    sources = []
    seen = set()

    for item in contexts:
        source = item.get("source") or "unknown"
        page = item.get("page", "?")
        key = f"{source}|{page}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page})
        if len(sources) >= limit:
            break

    return sources


def get_retrieval_debug(question: str, chat_history: Optional[List[dict]] = None, fast_mode: bool = False) -> Dict[str, Any]:
    """
    Debug tool: Returns detailed retrieval information for a question.
    Useful for understanding why certain documents/chunks were chosen (or not).
    Includes rewritten query, top chunks with scores, sources, etc.
    """
    if not _initialized:
        init_rag()

    effective_k = 8 if fast_mode else INITIAL_RETRIEVE_K
    effective_final_k = 4 if fast_mode else FINAL_TOP_K
    do_rewrite = not fast_mode and USE_QUERY_REWRITING
    do_rerank = not fast_mode and USE_RERANKER

    rewritten = rewrite_query(question, chat_history) if do_rewrite else question

    # Initial retrieval
    raw_docs = _retriever.invoke(rewritten)

    # Always light boost even for debug
    docs = light_keyword_boost_reorder(rewritten, raw_docs, top_k=len(raw_docs))

    # Get scores using reranker (even if not using for final)
    scored = []
    if _embeddings is not None and docs:
        scored = rerank_documents(rewritten, docs, _embeddings, top_k=len(docs), return_scores=True)

    # Apply same protection as normal path, using raw for top to keep variety
    hybrid_top = raw_docs[:5]
    if do_rerank and _embeddings is not None and len(docs) > effective_final_k:
        docs = rerank_documents(rewritten, docs, _embeddings, top_k=effective_final_k)
    else:
        docs = docs[:effective_final_k]

    # Relevance guard
    if USE_RELEVANCE_GUARD and _embeddings is not None and docs:
        docs = filter_relevant_chunks(
            rewritten, docs, _embeddings,
            min_score=MIN_RELEVANCE_SCORE, min_chunks=MIN_RELEVANT_CHUNKS
        )

    # Build debug info
    chunks_info = []
    for i, doc in enumerate(docs):
        info = {
            "rank": i + 1,
            "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", "?"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
        # Try to find score if available
        for score, sdoc in scored[:effective_final_k]:
            if sdoc.page_content[:100] == doc.page_content[:100]:
                info["score"] = round(float(score), 4)
                break
        chunks_info.append(info)

    return {
        "original_question": question,
        "rewritten_query": rewritten if do_rewrite else None,
        "fast_mode": fast_mode,
        "num_retrieved_initial": len(_retriever.invoke(rewritten)) if _retriever else 0,
        "top_chunks": chunks_info,
        "effective_k": effective_k,
        "effective_final_k": effective_final_k,
        "used_rewrite": do_rewrite,
        "used_rerank": do_rerank,
    }


async def _stream_single_answer(
    question: str,
    chat_history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    if _is_offtopic_query(question) or _is_unsupported_future_query(question):
        print(f"[Refusal] Off-topic / out-of-corpus query (stream): {question[:100]}")
        yield NO_INFO_ANSWER
        return

    if is_nckh_lecturer_query(question):
        extracted = try_extract_nckh_lecturers(question, [], _chunks_cache)
        if extracted:
            print(f"[Extract] NCKH lecturer stream ({extracted[0].count('**') // 2} entries, pre-retrieval)")
            yield extracted[0]
            return

    extracted = _try_pre_retrieval_extract(question)
    if extracted:
        print("[Extract] Structured facts (pre-retrieval)")
        yield extracted[0]
        return

    retrieval_query, docs, refused, reason, logd = _retrieve_filtered_docs(
        question, chat_history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    )
    if refused:
        extracted = _maybe_extract_nckh_facts(question, [])
        if extracted:
            print("[Extract] NCKH facts (refusal fallback)")
            yield extracted[0]
            return
        extracted = _maybe_extract_policy_facts(question, [])
        if extracted:
            print("[Extract] Policy facts (refusal fallback)")
            yield extracted[0]
            return
        _log_refusal(retrieval_query, docs, reason, logd)
        yield NO_INFO_ANSWER
        return

    extracted = try_extract_nckh_lecturers(question, docs, _chunks_cache)
    if extracted:
        print(f"[Extract] NCKH lecturer table from PDF chunks ({extracted[0].count('**') // 2} entries)")
        yield extracted[0]
        return

    extracted = _maybe_extract_nckh_facts(question, docs)
    if extracted:
        print("[Extract] NCKH facts from PDF chunks")
        yield extracted[0]
        return

    extracted = _maybe_extract_policy_facts(question, docs)
    if extracted:
        print("[Extract] Policy facts from PDF chunks")
        yield extracted[0]
        return

    context = format_context(docs, question)
    context_ms = round((asyncio.get_running_loop().time() - request_started) * 1000 - retrieval_ms, 1)
    max_tokens = _output_token_budget(fast_mode)

    if _use_gemini():
        try:
            _log_llm_backend("stream", question)
            async for chunk in _stream_with_gemini(context, question, max_tokens):
                yield chunk
            return
        except Exception as e:
            print(f"[Gemini] Streaming failed, falling back to Ollama: {e}")

    _log_llm_backend("stream-fallback" if LLM_PROVIDER == "gemini" else "stream", question)
    prompt = build_prompt()
    intent = _query_intent(question)
    tuyensinh_fast = fast_mode and intent.get("tuyensinh") and not intent["nckh"] and not intent["hocphan"]
    stream_opts = {
        "num_predict": max_tokens,
        "num_ctx": 2048 if tuyensinh_fast else (1024 if fast_mode else NUM_CTX),
        "num_gpu": OLLAMA_NUM_GPU,
        "num_thread": OLLAMA_NUM_THREAD,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    llm = (_llm if not fast_mode and _llm else None) or OllamaLLM(
        model=MODEL,
        timeout=LLM_TIMEOUT,
        options=stream_opts,
    )
    chain = prompt | llm | StrOutputParser()
    async for chunk in _stream_ollama_chain(
        chain, {"context": context, "question": question}, label="LLM"
    ):
        yield chunk


async def stream_answer_with_context(
    question: str,
    chat_history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream an answer through the unified pure-RAG pipeline."""
    if PURE_RAG_MODE:
        result = _ask_single_question(question, chat_history, False, use_hybrid=use_hybrid, filter_source=filter_source)
        yield result.get("answer", NO_INFO_ANSWER)
        return

    sub_questions = [question] if PURE_RAG_MODE else _questions_for_pipeline(question, fast_mode)
    if len(sub_questions) > 1:
        print(f"[Multi-Q] Split into {len(sub_questions)} sub-questions")
        for idx, sub_q in enumerate(sub_questions, 1):
            if idx > 1:
                yield "\n\n"
            yield f"**{idx}. {sub_q}**\n"
            async for chunk in _stream_single_answer(
                sub_q, chat_history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
            ):
                yield chunk
        return

    async for chunk in _stream_single_answer(
        question, chat_history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    ):
        yield chunk


def _ask_single_question(
    question: str,
    history: Optional[List[dict]] = None,
    fast_mode: bool = False,
    use_hybrid: Optional[bool] = None,
    filter_source: Optional[str] = None,
) -> Dict[str, Any]:
    if PURE_RAG_MODE:
        retrieval_query, docs, refused, reason, logd = _retrieve_filtered_docs(
            question, history, False, use_hybrid=use_hybrid, filter_source=filter_source
        )
        _write_rag_audit(question, retrieval_query, docs, logd, refused, reason)
        if refused:
            _log_refusal(retrieval_query, docs, reason, logd)
            return {"answer": NO_INFO_ANSWER, "sources": [], "mode": "pure_rag"}
        try:
            answer = _clean_final_answer(
                _generate_answer_from_docs(question, docs, fast_mode=True), question
            )
        except Exception as exc:
            print(f"[LLM] Pure RAG answer generation failed: {exc!r}")
            return {"answer": NO_INFO_ANSWER, "sources": [], "mode": "pure_rag"}
        sources = [] if answer == NO_INFO_ANSWER else _docs_to_sources(docs)
        return {"answer": answer, "sources": sources, "mode": "pure_rag"}

    if _is_offtopic_query(question) or _is_unsupported_future_query(question):
        print("[Refusal] Off-topic / out-of-corpus query")
        return _offtopic_refusal_response(question, fast_mode)

    if is_nckh_lecturer_query(question):
        extracted = try_extract_nckh_lecturers(question, [], _chunks_cache)
        if extracted:
            answer, sources = extracted
            print(f"[Extract] NCKH lecturer table ({answer.count('**') // 2} entries, pre-retrieval)")
            return {"answer": answer, "sources": sources, "mode": "rag"}

    extracted = _try_pre_retrieval_extract(question)
    if extracted:
        answer, sources = extracted
        print("[Extract] Structured facts (pre-retrieval)")
        return {"answer": answer, "sources": sources, "mode": "rag"}

    retrieval_query, docs, refused, reason, logd = _retrieve_filtered_docs(
        question, history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    )
    if refused:
        if refused:
            extracted = _maybe_extract_nckh_facts(question, [])
            if extracted:
                answer, sources = extracted
                print("[Extract] NCKH facts (refusal fallback)")
                return {"answer": answer, "sources": sources, "mode": "rag"}
            extracted = _maybe_extract_policy_facts(question, [])
            if extracted:
                answer, sources = extracted
                print("[Extract] Policy facts (refusal fallback)")
                return {"answer": answer, "sources": sources, "mode": "rag"}
            _log_refusal(retrieval_query, docs, reason, logd)
            return {"answer": NO_INFO_ANSWER, "sources": [], "mode": response_mode}

    extracted = try_extract_nckh_lecturers(question, docs, _chunks_cache)
    if extracted:
        answer, sources = extracted
        print(f"[Extract] NCKH lecturer table from PDF chunks")
        return {"answer": answer, "sources": sources, "mode": "rag"}

    extracted = _maybe_extract_nckh_facts(question, docs)
    if extracted:
        answer, sources = extracted
        print("[Extract] NCKH facts from PDF chunks")
        return {"answer": answer, "sources": sources, "mode": "rag"}

    extracted = _maybe_extract_policy_facts(question, docs)
    if extracted:
        answer, sources = extracted
        print("[Extract] Policy facts from PDF chunks")
        return {"answer": answer, "sources": sources, "mode": "rag"}

    try:
        answer = _clean_final_answer(
            _generate_answer_from_docs(question, docs, fast_mode=fast_mode),
            question,
        )
    except Exception as exc:
        print(f"[LLM] Answer generation failed: {exc}")
        return {"answer": NO_INFO_ANSWER, "sources": [], "mode": "rag"}
    sources = [] if answer == NO_INFO_ANSWER else _docs_to_sources(docs)
    return {"answer": answer, "sources": sources, "mode": "rag"}


def ask_question(question: str, use_hybrid: Optional[bool] = None, filter_source: Optional[str] = None, history: Optional[List[dict]] = None, fast_mode: bool = False) -> Dict[str, Any]:
    sub_questions = [question] if PURE_RAG_MODE else _questions_for_pipeline(question, fast_mode)
    if len(sub_questions) > 1:
        print(f"[Multi-Q] Split into {len(sub_questions)} sub-questions (parallel)")
        workers = min(MAX_PARALLEL_QUESTIONS, len(sub_questions))
        results_by_idx: Dict[int, Dict[str, Any]] = {}

        def _run_one(idx: int, sub_q: str) -> tuple[int, Dict[str, Any]]:
            return idx, _ask_single_question(
                sub_q, history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, i, sq) for i, sq in enumerate(sub_questions, 1)]
            for fut in as_completed(futures):
                idx, result = fut.result()
                results_by_idx[idx] = result

        blocks: List[str] = []
        source_lists: List[List[Dict]] = []
        for idx in sorted(results_by_idx):
            sub_q = sub_questions[idx - 1]
            result = results_by_idx[idx]
            blocks.append(f"**{idx}. {sub_q}**\n{result.get('answer', NO_INFO_ANSWER)}")
            source_lists.append(result.get("sources", []))
        modes = [results_by_idx[i].get("mode", "rag") for i in sorted(results_by_idx)]
        combined_mode = "rag"
        return {
            "answer": "\n\n".join(blocks),
            "sources": _merge_sources(source_lists),
            "mode": combined_mode,
        }

    return _ask_single_question(
        question, history, fast_mode, use_hybrid=use_hybrid, filter_source=filter_source
    )

def get_status():
    doc_count = len(list_documents()) if Path(DATA_DIR).exists() else 0
    return {
        "initialized": _initialized,
        "provider": LLM_PROVIDER,
        "model": GEMINI_MODEL if _use_gemini() else MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "data_dir": DATA_DIR,
        "hybrid": USE_HYBRID,
        "min_relevance_score": MIN_RELEVANCE_SCORE,
        "documents_count": doc_count,
        "no_info_answer": NO_INFO_ANSWER,
        "index_status": _index_status,
    }

def get_index_status():
    global _index_status, _index_error
    return {
        "status": _index_status,
        "error": _index_error
    }

def set_index_status(status: str, error: str = None):
    global _index_status, _index_error
    _index_status = status
    _index_error = error

def _refresh_retriever():
    """Reload vectorstore, chunks and retriever from disk after index changes.
    Called after hot add/remove to make new content immediately searchable.
    """
    global _retriever, _vectorstore, _chunks_cache
    if not _embeddings:
        return
    index_path = Path(INDEX_DIR)
    try:
        _vectorstore = load_faiss_index(_embeddings, index_path)
        cache_path = index_path / "chunks_cache.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            from langchain_core.documents import Document
            _chunks_cache = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in cached]
        _retriever = get_retriever(
            _vectorstore,
            k=INITIAL_RETRIEVE_K,
            use_hybrid=USE_HYBRID,
            chunks_for_bm25=_chunks_cache or [],
        )
        print("[rag_service] Retriever reloaded after document change.")
    except Exception as e:
        print(f"[rag_service] Could not refresh retriever ({e}), will re-init on next query.")


def add_documents(file_paths: List[str]):
    if not file_paths:
        return
    set_index_status("indexing")
    try:
        if not _initialized:
            init_rag()
        from rag_backend import add_or_update_files
        add_or_update_files(file_paths)
        _refresh_retriever()
        set_index_status("ready")
    except Exception as e:
        set_index_status("error", str(e))
        print(f"[rag_service] add_documents failed: {e}")
        raise

def _clean_final_answer(ans: str, question: str = "", response_mode: str = "rag") -> str:
    """Normalize refusal phrases; keep substantive answers when the model mixed refusal with content."""
    text = (ans or "").strip()
    refusals = [
        NO_INFO_ANSWER,
        "Tôi không biết dựa trên tài liệu.",
        "Tôi không biết dự trên tài liệu.",
        "I don't know based on the document.",
    ]
    for refusal in refusals:
        if refusal in text:
            before = text.split(refusal)[0].strip()
            if before and len(before) > 5:
                return before
            return NO_INFO_ANSWER

    if _is_offtopic_query(question) or _is_arithmetic_or_trivia_query(question):
        return NO_INFO_ANSWER

    if re.search(r"[\u4e00-\u9fff]", text) and not _query_in_corpus_domain((question or "").lower()):
        return NO_INFO_ANSWER

    lowered = text.lower()
    no_info_markers = [
        "không có thông tin",
        "không được đề cập",
        "không liên quan",
        "không có trong tài liệu",
        "không có trong context",
        "không có trong thông tin",
    ]
    intent = _query_intent(question)
    if not intent["yes_no"] and any(marker in lowered for marker in no_info_markers):
        return NO_INFO_ANSWER

    return text


def list_documents() -> List[Dict]:
    data_dir = Path(DATA_DIR)
    docs = []
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            if f.is_file():
                docs.append({"name": f.name, "size": f.stat().st_size})
    return docs

def delete_document(filename: str) -> bool:
    path = Path(DATA_DIR) / filename
    if path.exists():
        path.unlink()
        return True
    return False

def remove_documents(file_names: List[str]):
    if not file_names:
        return
    set_index_status("indexing")
    try:
        if not _initialized:
            init_rag()
        from rag_backend import remove_documents as rb_remove
        rb_remove(file_names)
        _refresh_retriever()
        set_index_status("ready")
    except Exception as e:
        set_index_status("error", str(e))
        print(f"[rag_service] remove_documents failed: {e}")
        raise


def rebuild_index():
    set_index_status("indexing")
    try:
        res = init_rag(force=True)
        set_index_status("ready")
        return res
    except Exception as e:
        set_index_status("error", str(e))
        print(f"[rag_service] rebuild_index failed: {e}")
        raise

def get_data_dir() -> str:
    """Return the configured data directory for documents."""
    return DATA_DIR


def search_documents(query: str = "", mode: str = "name", limit: int = 100) -> List[Dict]:
    """Advanced search over documents.

    mode: 'name' (filename match) or 'semantic' (RAG similarity)
    """
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return []

    q = (query or "").lower().strip()

    if not q:
        # return all
        docs = []
        for f in sorted(data_dir.iterdir()):
            if f.is_file():
                docs.append({"name": f.name, "size": f.stat().st_size})
        return docs[:limit]

    if mode == "name":
        docs = []
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and q in f.name.lower():
                docs.append({"name": f.name, "size": f.stat().st_size})
        return docs[:limit]

    # semantic mode - use retriever
    if not _initialized:
        try:
            init_rag()
        except Exception:
            pass

    if _retriever is None:
        # fallback to name
        docs = []
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and q in f.name.lower():
                docs.append({"name": f.name, "size": f.stat().st_size})
        return docs[:limit]

    try:
        retrieved = _retriever.invoke(query)[: limit * 3]
    except Exception:
        retrieved = []

    seen = {}
    results = []
    for doc in retrieved:
        name = doc.metadata.get("source", "")
        if not name or name in seen:
            continue
        seen[name] = True

        snippet = doc.page_content[:180].replace("\n", " ").strip()
        if len(doc.page_content) > 180:
            snippet += "..."

        # try to get real size
        try:
            size = (data_dir / name).stat().st_size
        except Exception:
            size = 0

        results.append({
            "name": name,
            "size": size,
            "snippet": snippet,
            "match_type": "semantic"
        })
        if len(results) >= limit:
            break

    return results
