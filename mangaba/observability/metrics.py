"""Cost + usage metrics (Fase 3).

CrewAI AMP offers actionable insights / token aggregation. Mangaba gets a
tiny offline cost estimator plus a Prometheus-compatible text exporter so
existing dashboards can scrape crew runs without new dependencies.

Example::

    from mangaba.observability.metrics import estimate_cost, prometheus_text

    cost = estimate_cost({"prompt_tokens": 1000, "completion_tokens": 500}, model="gpt-4o-mini")
    print(prometheus_text("my_crew", {"total_tokens": 1500}, cost_usd=cost))
"""

from __future__ import annotations

from typing import Dict

# USD per 1k tokens (prompt, completion). Rough public pricing, 2026.
_PRICE_PER_1K: Dict[str, tuple] = {
    "default": (0.0015, 0.002),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gemini-2.5-flash": (0.0003, 0.0025),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3.5-sonnet": (0.003, 0.015),
}


def estimate_cost(token_usage: Dict, model: str = "default") -> float:
    """Estimate USD cost from a token_usage dict. Never raises."""
    try:
        prompt = int(token_usage.get("prompt_tokens", 0) or 0)
        completion = int(token_usage.get("completion_tokens", 0) or 0)
    except Exception:
        return 0.0
    key = model if model in _PRICE_PER_1K else "default"
    p_price, c_price = _PRICE_PER_1K[key]
    return round(prompt / 1000 * p_price + completion / 1000 * c_price, 6)


def prometheus_text(crew_id: str, token_usage: Dict, cost_usd: float = 0.0, duration_s: float = 0.0) -> str:
    """Render crew metrics in Prometheus exposition format."""
    safe = "".join(c if c.isalnum() or c in "_:" else "_" for c in str(crew_id))
    lines = [
        "# HELP mangaba_tokens_total Total tokens used by the crew.",
        "# TYPE mangaba_tokens_total counter",
        f'mangaba_tokens_total{{crew="{safe}"}} {int(token_usage.get("total_tokens", 0) or 0)}',
        "# HELP mangaba_cost_usd Estimated USD cost.",
        "# TYPE mangaba_cost_usd gauge",
        f'mangaba_cost_usd{{crew="{safe}"}} {float(cost_usd)}',
        "# HELP mangaba_duration_seconds Crew duration in seconds.",
        "# TYPE mangaba_duration_seconds gauge",
        f'mangaba_duration_seconds{{crew="{safe}"}} {float(duration_s)}',
    ]
    return "\n".join(lines) + "\n"


__all__ = ["estimate_cost", "prometheus_text"]
