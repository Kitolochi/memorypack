"""Split text into semantic chunks at sentence boundaries."""

from __future__ import annotations

import nltk

from memorypack.models import Chunk
from memorypack.tokencount import estimate_tokens

_nltk_ready = False


def _ensure_nltk() -> None:
    global _nltk_ready
    if not _nltk_ready:
        nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    _ensure_nltk()
    return nltk.sent_tokenize(text)


def chunk_text(
    text: str, source_file: str, target_tokens: int = 512, start_id: int = 0
) -> list[Chunk]:
    """Split text into chunks of approximately target_tokens at sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_id = start_id

    for para in paragraphs:
        # Check if this is a heading — keep it attached to next content
        if para.startswith("#"):
            current_sentences.append(para)
            current_tokens += estimate_tokens(para)
            continue

        sentences = _split_sentences(para)
        for sentence in sentences:
            sent_tokens = estimate_tokens(sentence)

            if current_tokens + sent_tokens > target_tokens and current_sentences:
                # Flush current chunk
                chunk_text_str = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text_str,
                        source_file=source_file,
                        token_count=estimate_tokens(chunk_text_str),
                    )
                )
                chunk_id += 1
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(
            Chunk(
                id=chunk_id,
                text=chunk_text_str,
                source_file=source_file,
                token_count=estimate_tokens(chunk_text_str),
            )
        )

    return chunks
