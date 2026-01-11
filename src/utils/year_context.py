"""Deterministic year context tracking utilities.
Goal: keep the most recently referenced year stable across turns without extra LLM calls.
"""

import re

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def infer_year_context(question: str, prev_year: str | None) -> str | None:
    """Infer the year context for the current turn.
    Rules (deterministic):
    - If the question explicitly mentions a year (e.g., 2017), use that.
    - Else if the question says "N years earlier/later" and prev_year exists, shift prev_year.
    - Else if prev_year exists, carry it forward (handles "that year/that amount/these").
    - Else return None.
    """
    q = question.lower()

    # 1) Explicit year mention wins
    m = YEAR_RE.search(question)
    if m:
        return m.group(0)

    # 2) Relative references (common in dataset)
    if prev_year is not None and prev_year.isdigit():
        base = int(prev_year)

        # e.g. "two years earlier", "3 years earlier"
        m_earlier = re.search(r"\b(\d+)\s+years?\s+earlier\b", q)
        if m_earlier:
            return str(base - int(m_earlier.group(1)))

        # e.g. "two years later", "3 years later"
        m_later = re.search(r"\b(\d+)\s+years?\s+later\b", q)
        if m_later:
            return str(base + int(m_later.group(1)))

        # e.g. "next year", "the following year"
        if re.search(r"\b(next year|following year)\b", q):
            return str(base + 1)

        # e.g. "previous year", "prior year"
        if re.search(r"\b(previous year|prior year)\b", q):
            return str(base - 1)

    # 3) Carry-forward year context
    return prev_year
