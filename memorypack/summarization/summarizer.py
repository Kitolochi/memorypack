"""Hierarchical BART summarization."""

from __future__ import annotations

import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*max_length.*input_length.*")
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")

from transformers import pipeline as hf_pipeline

from memorypack.config import PipelineConfig
from memorypack.models import Cluster
from memorypack.tokencount import estimate_tokens


def load_summarizer(config: PipelineConfig):
    """Load the BART summarization pipeline."""
    return hf_pipeline(
        "summarization",
        model=config.summarization_model,
        device=(-1 if config.device == "cpu" else 0),
        framework="pt",
    )


def _chunk_for_bart(text: str, max_tokens: int = 900) -> list[str]:
    """Split text into segments that fit BART's 1024-token input limit.

    Uses ~900 tokens to leave headroom.
    """
    sentences = text.replace("\n", " ").split(". ")
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            segments.append(". ".join(current) + ".")
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        segments.append(". ".join(current) + ".")

    return segments if segments else [text[:3000]]


def summarize_cluster(
    cluster: Cluster, summarizer, config: PipelineConfig
) -> str:
    """Hierarchically summarize a cluster's chunks.

    If combined text exceeds BART's input limit, summarize segments
    first, then summarize the summaries.
    """
    combined = " ".join(c.text for c in cluster.chunks)
    total_tokens = estimate_tokens(combined)

    if total_tokens <= 900:
        # Fits in one pass
        result = summarizer(
            combined,
            max_length=config.summary_max_tokens,
            min_length=30,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"]

    # Hierarchical: summarize segments, then summarize summaries
    segments = _chunk_for_bart(combined)
    intermediate_summaries: list[str] = []

    for segment in segments:
        result = summarizer(
            segment,
            max_length=config.summary_max_tokens,
            min_length=20,
            do_sample=False,
            truncation=True,
        )
        intermediate_summaries.append(result[0]["summary_text"])

    # Combine intermediate summaries
    merged = " ".join(intermediate_summaries)
    if estimate_tokens(merged) > 900:
        # Need another round
        segments = _chunk_for_bart(merged)
        final_parts = []
        for segment in segments:
            result = summarizer(
                segment,
                max_length=config.summary_max_tokens,
                min_length=20,
                do_sample=False,
                truncation=True,
            )
            final_parts.append(result[0]["summary_text"])
        return " ".join(final_parts)

    result = summarizer(
        merged,
        max_length=config.summary_max_tokens,
        min_length=30,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


def generate_overview(
    clusters: list[Cluster], summarizer, config: PipelineConfig
) -> str:
    """Generate a meta-summary from all cluster summaries."""
    combined = " ".join(
        f"{c.label}: {c.summary}" for c in clusters if c.summary
    )

    if not combined.strip():
        return "No content to summarize."

    if estimate_tokens(combined) > 900:
        segments = _chunk_for_bart(combined)
        parts = []
        for segment in segments:
            result = summarizer(
                segment,
                max_length=config.overview_max_tokens,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
            parts.append(result[0]["summary_text"])
        combined = " ".join(parts)

    result = summarizer(
        combined,
        max_length=config.overview_max_tokens,
        min_length=40,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]
