"""Chunk-scoped JSON extraction for PURE_RAG list requests only."""
import json
import re
import unicodedata
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple


def _fold_answer_shape_text(value: str) -> str:
    """Compare Vietnamese question form independently of diacritics and case."""
    folded = unicodedata.normalize("NFD", value or "").casefold()
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", folded.replace("\u0111", "d")).strip()


def classify_answer_shape(question: str, chunks: Sequence[Any]) -> Dict[str, Any]:
    """Classify requested answer form before any table/list extraction."""
    normalized = _fold_answer_shape_text(question)
    numbered = sum(numbered_row_count(chunk.page_content) for chunk in chunks)
    explicit_list = ("list", "enumerate", "liet ke", "danh sach", "nhung ai", "gom nhung gi")
    single_markers = (
        "email cua", "so dien thoai cua", "what is", "bao nhieu", "bao lau",
        "khi nao", "o dau", "ngay nao", "la ai", "thoi han", "thoi gian",
    )
    # Quantity/duration/where forms take precedence over the leading Vietnamese
    # word "co"; e.g. "Co bao nhieu...?" is not a yes/no question.
    if any(term in normalized for term in single_markers):
        return {"answer_shape": "single", "classification_method": "question_structure", "classification_confidence": 0.90, "numbered_rows_detected": numbered}
    if re.search(r"^(co|khong|is|are|does|do)\b", normalized) or re.search(r"\b(co|khong|duoc khong)\s*\??$", normalized):
        return {"answer_shape": "boolean", "classification_method": "question_structure", "classification_confidence": 0.95, "numbered_rows_detected": numbered}
    if any(term in normalized for term in ("giai thich", "vi sao", "tai sao", "why", "how does")):
        return {"answer_shape": "explanation", "classification_method": "question_structure", "classification_confidence": 0.90, "numbered_rows_detected": numbered}
    if any(term in normalized for term in explicit_list):
        return {"answer_shape": "list", "classification_method": "question_structure", "classification_confidence": 0.95, "numbered_rows_detected": numbered}
    if numbered >= 2:
        return {"answer_shape": "list", "classification_method": "question_context", "classification_confidence": 0.80, "numbered_rows_detected": numbered}
    return {"answer_shape": "uncertain", "classification_method": "needs_llm", "classification_confidence": 0.0, "numbered_rows_detected": numbered}
def numbered_row_count(text: str) -> int:
    return len(re.findall(r"(?:^|\n|\|)\s*\d{1,3}\s+[A-Za-zÀ-ỹ]", text))


def parse_records(raw: str) -> List[Dict[str, Any]]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


async def extract_records_by_chunk(
    question: str,
    chunks: Sequence[Any],
    generate_json: Callable[[str], Awaitable[str]],
    max_attempts: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for chunk in chunks:
        expected = numbered_row_count(chunk.page_content)
        prompt = (
            "Extract every numbered record from this DOCUMENT CHUNK as strict JSON only: "
            "[{\"name\": string, \"email\": string|null, \"details\": string|null}]. "
            "Do not summarize and do not omit records.\n\nDOCUMENT CHUNK:\n"
            + chunk.page_content
        )
        parsed: List[Dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            parsed = parse_records(await generate_json(prompt))
            if len(parsed) >= expected:
                break
        for record in parsed:
            record = dict(record)
            record.update({
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "chunk_id": chunk.metadata.get("chunk_id"),
            })
            records.append(record)
        audit.append({
            "source": chunk.metadata.get("source"), "page": chunk.metadata.get("page"),
            "chunk_id": chunk.metadata.get("chunk_id"), "expected_numbered_rows": expected,
            "extracted_records": len(parsed), "attempts": attempt,
        })
    unique: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = re.sub(r"\s+", " ", str(record.get("name", "")).strip().casefold())
        if key and key not in seen:
            seen.add(key)
            unique.append(record)
    return unique, audit
