"""Keyword extraction helpers for brief indexing mode."""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import List, Sequence

from .content_extract_service import extract_head

BRIEF_CHAR_LIMIT = 4000
BRIEF_KEYWORD_LIMIT = 20

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "with",
    "this",
    "that",
    "from",
    "have",
    "will",
    "should",
    "must",
    "need",
    "user",
    "users",
    "data",
    "when",
    "your",
    "their",
    "about",
    "into",
    "which",
    "within",
    "where",
    "while",
    "there",
    "only",
    "each",
    "more",
    "than",
    "also",
    "such",
    "shall",
    "can",
    "may",
    "our",
    "per",
    "any",
    "all",
    "like",
    "been",
    "over",
    "ensure",
    "including",
    "include",
}
_STOPWORDS_ZH = {"的", "了", "和", "或", "及", "需要", "支持", "功能", "用户", "系统"}


def summarize_keywords(
    path: Path,
    *,
    char_limit: int = BRIEF_CHAR_LIMIT,
    limit: int = BRIEF_KEYWORD_LIMIT,
) -> list[str]:
    """Return a list of high-frequency keywords extracted from *path*."""

    snippet = extract_head(path, char_limit=char_limit)
    if not snippet:
        return []
    return _extract_keywords(snippet, limit)


def _extract_keywords(text: str, limit: int) -> list[str]:
    if not text:
        return []
    display_map: dict[str, str] = {}
    counter: Counter[str] = Counter()

    for match in _WORD_RE.finditer(text):
        raw = match.group(0)
        key = raw.lower()
        if len(key) < 3 or key in _STOPWORDS:
            continue
        counter[key] += 1
        display_map.setdefault(key, raw)

    for match in _CJK_RE.finditer(text):
        token = match.group(0)
        if token in _STOPWORDS_ZH:
            continue
        counter[token] += 1
        display_map.setdefault(token, token)

    if not counter:
        return []
    keywords: list[str] = []
    for key, _ in counter.most_common(limit):
        keywords.append(display_map.get(key, key))
    return keywords

