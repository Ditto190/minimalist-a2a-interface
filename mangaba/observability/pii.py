"""PII redaction (Fase 3 — CrewAI AMP parity, lite).

CrewAI AMP offers runtime PII masking. Mangaba gets a dependency-free
version: :func:`redact_pii` scrubs common patterns (email, phones, CPF/CNPJ,
credit cards, API keys) and can be applied to prompts, outputs or logs.

Example::

    from mangaba.observability.pii import redact_pii

    clean = redact_pii("contato joao@empresa.com / CPF 123.456.789-00")
"""

from __future__ import annotations

import re
from typing import Dict, Pattern

_PATTERNS: Dict[str, Pattern] = {
    "api_key": re.compile(r"\b(sk-[A-Za-z0-9-_]{8,}|gsk_[A-Za-z0-9]{8,}|AIza[A-Za-z0-9-_]{8,})\b"),
    "bearer": re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "cpf": re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
    "cnpj": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
    "phone_br": re.compile(r"\(?\d{2}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

_REPLACEMENTS = {
    "email": "[EMAIL]",
    "phone_br": "[PHONE]",
    "cpf": "[CPF]",
    "cnpj": "[CNPJ]",
    "credit_card": "[CARD]",
    "api_key": "[API_KEY]",
    "bearer": "Bearer [REDACTED]",
}


def redact_pii(text: str, extra_patterns: Dict[str, str] | None = None) -> str:
    """Replace PII matches with placeholders. Never raises."""
    if not text:
        return text
    out = str(text)
    for name, pattern in _PATTERNS.items():
        try:
            out = pattern.sub(_REPLACEMENTS[name], out)
        except Exception:
            continue
    for name, replacement in (extra_patterns or {}).items():
        try:
            out = re.sub(name, replacement, out)
        except Exception:
            continue
    return out


def contains_pii(text: str) -> bool:
    """True when any known PII pattern matches."""
    if not text:
        return False
    return any(p.search(str(text)) for p in _PATTERNS.values())


__all__ = ["redact_pii", "contains_pii"]
