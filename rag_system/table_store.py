"""Transactional SQLite store for numbered PDF tables."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1"
ROW_RE = re.compile(r"(?m)(?:^|\n)\s*(\d{1,4})\s+(?=[A-ZÀ-ỴĐ])")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normal(text: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text or "").split())


def _headers(text: str) -> list[str]:
    for line in text.splitlines()[:12]:
        words = re.split(r"\s{2,}|\t", line.strip())
        if len(words) >= 2 and any(w.casefold() in {"stt", "email", "họ", "tên"} for w in words):
            return words
    return ["row_number", "raw_text"]


def _is_footer_line(line: str) -> bool:
    folded = _normal(line).casefold()
    if not folded:
        return False
    footer_markers = (
        "pho truong khoa",
        "truong khoa",
        "da ky",
        "cong hoa xa hoi chu nghia viet nam",
        "bo giao duc",
        "truong dai hoc",
        "so:",
        "file:",
        "trang:",
        "tai lieu:",
        "ths.",
        "ts.",
        "pgs.",
        "gs.",
    )
    if any(marker in folded for marker in footer_markers):
        return True
    words = folded.split()
    if line.strip().isupper() and len(words) <= 10:
        return True
    if len(words) <= 4 and sum(char.isdigit() for char in line) >= 2:
        return True
    return False


def _trim_row_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    cut = len(lines)
    for idx, line in enumerate(lines):
        if idx > 0 and _is_footer_line(line):
            cut = idx
            break
    return "\n".join(lines[:cut]).strip()


def _consolidate(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(row["row_number"], []).append(row)
    rows: list[dict[str, Any]] = []
    conflict = False
    for number, items in grouped.items():
        normalized = [_normal(item["raw_text"]) for item in items]
        best = max(items, key=lambda item: len(_normal(item["raw_text"])))
        provenance = [{"page": item["page"], "chunk_id": item["chunk_id"]} for item in items]
        if len(set(normalized)) == 1:
            status = "benign_duplicate"
        elif all(value in _normal(best["raw_text"]) or _normal(best["raw_text"]) in value for value in normalized):
            status = "merged_overlap"
        else:
            status = "conflict"
            conflict = True
        best = dict(best)
        best["cells"] = dict(best["cells"])
        best["cells"]["provenance"] = provenance
        best["cells"]["merge_status"] = status
        rows.append(best)
    return sorted(rows, key=lambda row: row["row_number"]), conflict


def extract_tables(chunks: Iterable[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Any]] = {}
    for chunk in chunks:
        if ROW_RE.search(chunk.page_content or ""):
            key = (str(chunk.metadata.get("source", "")), str(chunk.metadata.get("doc_number", chunk.metadata.get("doc_title", ""))))
            groups.setdefault(key, []).append(chunk)
    tables: list[dict[str, Any]] = []
    for (source, marker), group in groups.items():
        group = sorted(group, key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_id", 0)))
        headers = _headers(group[0].page_content)
        table_id = _hash(source + "|" + marker + "|" + "|".join(headers))[:24]
        candidates: list[dict[str, Any]] = []
        for chunk in group:
            matches = list(ROW_RE.finditer(chunk.page_content))
            for i, match in enumerate(matches):
                raw = chunk.page_content[match.start() : matches[i + 1].start() if i + 1 < len(matches) else len(chunk.page_content)].strip()
                raw = _trim_row_text(raw)
                if not raw:
                    continue
                number = int(match.group(1))
                candidates.append({
                    "table_id": table_id,
                    "row_number": number,
                    "cells": {"row_number": number, "raw_text": raw},
                    "raw_text": raw,
                    "source": source,
                    "page": chunk.metadata.get("page"),
                    "chunk_id": chunk.metadata.get("chunk_id"),
                })
        rows, conflict = _consolidate(candidates)
        numbers = [row["row_number"] for row in rows]
        valid = bool(numbers) and not conflict and numbers == list(range(min(numbers), max(numbers) + 1))
        tables.append({
            "table_id": table_id,
            "source": source,
            "headers": headers,
            "first_page": min((row["page"] for row in rows), default=0),
            "last_page": max((row["page"] for row in rows), default=0),
            "rows": rows,
            "validation_status": "valid" if valid else "invalid",
            "validation_reason": None if valid else ("conflicting_duplicate_row" if conflict else "non_contiguous_row_number"),
            "content_hash": _hash("\n".join(row["raw_text"] for row in rows)),
        })
    return tables


def build_store(path: Path, chunks: Iterable[Any], index_hash: str) -> list[dict[str, Any]]:
    tables = extract_tables(chunks)
    con = sqlite3.connect(path)
    try:
        con.executescript(
            '''
            CREATE TABLE tables(
                table_id TEXT PRIMARY KEY,
                source TEXT,
                headers_json TEXT,
                first_page INTEGER,
                last_page INTEGER,
                row_count INTEGER,
                validation_status TEXT,
                schema_version TEXT,
                content_hash TEXT,
                index_hash TEXT
            );
            CREATE TABLE table_rows(
                table_id TEXT,
                row_number INTEGER,
                cells_json TEXT,
                raw_text TEXT,
                source TEXT,
                page INTEGER,
                chunk_id INTEGER,
                PRIMARY KEY(table_id,row_number)
            );
            '''
        )
        for table in tables:
            con.execute(
                "INSERT INTO tables VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    table["table_id"],
                    table["source"],
                    json.dumps(table["headers"], ensure_ascii=False),
                    table["first_page"],
                    table["last_page"],
                    len(table["rows"]),
                    table["validation_status"],
                    SCHEMA_VERSION,
                    table["content_hash"],
                    index_hash,
                ),
            )
            for row in table["rows"]:
                con.execute(
                    "INSERT INTO table_rows VALUES(?,?,?,?,?,?,?)",
                    (
                        row["table_id"],
                        row["row_number"],
                        json.dumps(row["cells"], ensure_ascii=False),
                        row["raw_text"],
                        row["source"],
                        row["page"],
                        row["chunk_id"],
                    ),
                )
        con.commit()
    finally:
        con.close()
    return tables


def valid_rows_for_sources(path: Path, sources: Iterable[str]) -> list[dict[str, Any]]:
    source_list = list(dict.fromkeys(str(value) for value in sources))
    if not path.exists() or not source_list:
        return []
    placeholders = ",".join("?" for _ in source_list)
    con = sqlite3.connect(path)
    try:
        query = f"""SELECT r.table_id,r.row_number,r.cells_json,r.raw_text,r.source,r.page,r.chunk_id
                    FROM table_rows r JOIN tables t ON t.table_id=r.table_id
                    WHERE t.validation_status='valid' AND r.source IN ({placeholders})
                    ORDER BY r.table_id,r.row_number"""
        rows = []
        for table_id, row_number, cells, raw_text, source, page, chunk_id in con.execute(query, source_list):
            rows.append({
                'table_id': table_id,
                'row_number': row_number,
                'cells': json.loads(cells),
                'raw_text': raw_text,
                'source': source,
                'page': page,
                'chunk_id': chunk_id,
            })
        return rows
    finally:
        con.close()
