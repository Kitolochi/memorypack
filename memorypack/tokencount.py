"""Fast token estimation and precise counting utilities."""

from __future__ import annotations

_WORD_TOKEN_RATIO = 1.3


def estimate_tokens(text: str) -> int:
    """Fast heuristic: words * 1.3."""
    return int(len(text.split()) * _WORD_TOKEN_RATIO)


def count_tokens_precise(text: str) -> int:
    """More precise count using character-based heuristic (chars / 4).

    Good enough for final stats without loading a tokenizer.
    """
    return max(1, len(text) // 4)
