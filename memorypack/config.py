"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """All tunable parameters for the compression pipeline."""

    # Chunking
    chunk_size: int = 512  # target tokens per chunk

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # cpu | cuda | mps

    # Deduplication
    dedup_threshold: float = 0.92  # cosine similarity threshold

    # Clustering
    max_clusters: int = 20
    min_clusters: int = 2

    # Summarization
    summarization_model: str = "facebook/bart-large-cnn"
    summary_max_tokens: int = 150  # per-cluster summary length
    overview_max_tokens: int = 250  # meta-summary length

    # Output
    compression_target: float = 6.0  # target compression ratio
    output_format: str = "single"  # single | multi
    topic: str = "Knowledge Base"


@dataclass
class PruneConfig:
    """Configuration for the prune command."""

    max_tokens: int = 0  # 0 = no budget limit
    min_importance: float = 0.0  # drop clusters below this score
    similarity_threshold: float = 0.80  # cosine threshold for near-duplicate detection
    merge_duplicates: bool = True  # merge near-duplicate clusters
    dry_run: bool = False  # print plan without modifying
    device: str = "cpu"  # cpu | cuda | mps
    embedding_model: str = "all-MiniLM-L6-v2"
