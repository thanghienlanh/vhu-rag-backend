"""Deterministic BM25 retrieval for Vietnamese queries with or without diacritics."""

from __future__ import annotations

from collections import Counter
from math import log
import re
import unicodedata
from typing import Any, Sequence


def tokenize(text: str) -> list[str]:
    """Fold Vietnamese diacritics so lexical matching is invariant to user input."""
    folded = unicodedata.normalize("NFD", (text or "").casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", folded.replace("\u0111", "d"))


class AccentInsensitiveBM25:
    """Small in-memory BM25 ranker that preserves the original document objects."""

    def __init__(self, documents: Sequence[Any], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies = [Counter(tokenize(getattr(doc, "page_content", ""))) for doc in self.documents]
        self.lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.document_frequency = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def score(self, query: str) -> list[float]:
        if not self.documents:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return [0.0] * len(self.documents)
        total_docs = len(self.documents)
        scores: list[float] = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                normalizer = tf + self.k1 * (1.0 - self.b + self.b * length / max(self.average_length, 1.0))
                score += idf * (tf * (self.k1 + 1.0) / normalizer)
            scores.append(score)
        return scores

    def search(self, query: str, k: int) -> list[Any]:
        scored = list(enumerate(self.score(query)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [self.documents[index] for index, score in scored[: max(0, int(k))] if score > 0.0]
