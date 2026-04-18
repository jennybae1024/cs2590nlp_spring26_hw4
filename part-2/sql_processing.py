"""
Shared SQL preprocessing / postprocessing for the T5 text-to-SQL pipeline.

Preprocess (training target form):
  - space-pad operators/punctuation so the tokenizer gets consistent tokens
  - lowercase everything
  - compress ATIS-style aliases (e.g. airport_service_1 -> as1) to shorten targets

Postprocess (inference output, produces strings matching raw ATIS GT form):
  - collapse spaces inside quoted literals and UPPERCASE their contents
    (DB values are stored uppercase, so this is required for Record EM/F1)
  - expand aliases back to their full form (as1 -> airport_service_1)
  - UPPERCASE SQL keywords
  - final whitespace collapse
"""

import re

ALIAS_MAP = {
    'flight_stop': 'fs',
    'flight_fare': 'ff',
    'flight_leg': 'fl',
    'fare_basis': 'fb',
    'fare': 'fr',
    'flight': 'f',
    'airport_service': 'as',
    'airport': 'ap',
    'airline': 'al',
    'aircraft': 'ac',
    'city': 'c',
    'code_description': 'cd',
    'class_of_service': 'cos',
    'compartment_class': 'cc',
    'date_day': 'dd',
    'days': 'd',
    'dual_carrier': 'dc',
    'equipment_sequence': 'es',
    'ground_service': 'gs',
    'food_service': 'fos',
    'month': 'mo',
    'restriction': 'r',
    'state': 'st',
    'time_interval': 'ti',
    'time_zone': 'tz',
}

SHORT_TO_LONG = {v: k for k, v in ALIAS_MAP.items()}

SQL_KEYWORDS = [
    'select', 'distinct', 'from', 'where', 'and', 'or', 'not', 'in', 'between',
    'like', 'is', 'null', 'min', 'max', 'count', 'sum', 'avg', 'as', 'on', 'join',
    'left', 'right', 'inner', 'outer', 'union', 'group', 'by', 'order', 'having',
    'limit', 'asc', 'desc', 'exists', 'all', 'any',
]


def preprocess_sql(query: str) -> str:
    """Lowercase, space-normalize operators, and compress table aliases."""
    # Pad multi-char operators first so they are space-isolated.
    query = re.sub(r'(<=|>=|!=)', r' \1 ', query)
    # Pad single-char operators, but skip chars that are part of a multi-char op
    # (`>=` must not become `> =`).
    query = re.sub(r"(?<![<>!])=(?!=)", ' = ', query)
    query = re.sub(r"<(?!=)", ' < ', query)
    query = re.sub(r">(?!=)", ' > ', query)
    query = re.sub(r"([(),'])", r' \1 ', query)
    query = query.lower()
    # Compress aliases. Longest table names first for hygiene (word-bounded
    # patterns make order correctness-irrelevant, but this is clearer).
    for long_name in sorted(ALIAS_MAP.keys(), key=len, reverse=True):
        short = ALIAS_MAP[long_name]
        query = re.sub(
            r'\b' + re.escape(long_name) + r'_(\d+)\b',
            short + r'\1',
            query,
        )
    query = re.sub(r'\s+', ' ', query).strip()
    return query


def _uppercase_quoted(m: "re.Match") -> str:
    return "'" + m.group(1).upper() + "'"


def postprocess_sql(query: str) -> str:
    """Inverse of preprocess_sql. Produces raw-ATIS-style SQL."""
    # Collapse spaces inside quoted strings and uppercase the literal content.
    # Non-greedy so multiple quoted segments are handled independently.
    query = re.sub(r"'\s*(.*?)\s*'", _uppercase_quoted, query)
    # Expand aliases (longest short-prefix first).
    for short in sorted(SHORT_TO_LONG.keys(), key=len, reverse=True):
        long_name = SHORT_TO_LONG[short]
        query = re.sub(
            r'\b' + re.escape(short) + r'(\d+)\b',
            long_name + r'_\1',
            query,
        )
    # Uppercase keywords (case-insensitive so we handle both cases).
    for kw in SQL_KEYWORDS:
        query = re.sub(r'\b' + kw + r'\b', kw.upper(), query, flags=re.IGNORECASE)
    query = re.sub(r'\s+', ' ', query).strip()
    return query
