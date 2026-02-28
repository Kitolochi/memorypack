"""Rule-based atomic fact extraction from text."""

from __future__ import annotations

import re

from memorypack.models import Cluster


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter (avoids loading NLTK again)."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_factual(sentence: str) -> bool:
    """Heuristic: is this sentence likely a factual statement?"""
    sentence = sentence.strip()
    if len(sentence) < 15:
        return False
    # Skip questions
    if sentence.endswith("?"):
        return False
    # Skip very vague or transitional statements
    vague_starts = ("this is", "it is important", "note that", "please",
                    "for example", "in other words", "basically",
                    "this approach", "this allows", "this process",
                    "this produces", "this makes", "this suggests",
                    "these allow", "these models", "several approaches",
                    "the process", "the key", "in practice")
    lower = sentence.lower()
    if any(lower.startswith(v) for v in vague_starts):
        return False
    # Skip sentences that are just context-dependent references
    if lower.startswith(("it ", "they ", "its ", "their ")):
        return False
    # Prefer sentences with proper nouns or technical terms (contain uppercase mid-sentence)
    words = sentence.split()
    if len(words) > 3:
        has_technical_term = any(w[0].isupper() and i > 0 for i, w in enumerate(words) if w[0].isalpha())
        has_number = any(c.isdigit() for c in sentence)
        if not has_technical_term and not has_number:
            return False
    return True


def _clean_fact(sentence: str) -> str:
    """Clean a sentence into a concise fact."""
    # Remove markdown formatting
    fact = re.sub(r"[*_`~]", "", sentence)
    # Remove leading bullets/numbers
    fact = re.sub(r"^[\s\-\d.)+]+", "", fact)
    # Strip heading markers
    fact = fact.lstrip("#").strip()
    # Capitalize first letter
    if fact and fact[0].islower():
        fact = fact[0].upper() + fact[1:]
    # Ensure ends with period
    if fact and not fact.endswith((".", "!", "?")):
        fact += "."
    return fact


def extract_facts(cluster: Cluster) -> list[str]:
    """Extract atomic facts from a cluster's chunks."""
    all_text = " ".join(c.text for c in cluster.chunks)
    sentences = _split_into_sentences(all_text)

    facts: list[str] = []
    seen: set[str] = set()

    for sentence in sentences:
        if not _is_factual(sentence):
            continue
        fact = _clean_fact(sentence)
        if not fact or len(fact) < 15:
            continue
        # Deduplicate within cluster
        normalized = fact.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            facts.append(fact)

    return facts
