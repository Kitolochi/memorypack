"""Data structures for the memorypack pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceFile:
    """A markdown file read from disk."""

    path: str
    raw_content: str
    metadata: dict = field(default_factory=dict)
    body: str = ""


@dataclass
class Chunk:
    """A semantic block of text (~512 tokens)."""

    id: int
    text: str
    source_file: str
    token_count: int
    embedding: list[float] | None = None
    is_duplicate: bool = False
    cluster_id: int = -1


@dataclass
class Cluster:
    """A group of semantically related chunks."""

    id: int
    chunks: list[Chunk] = field(default_factory=list)
    label: str = ""
    summary: str = ""
    facts: list[str] = field(default_factory=list)


@dataclass
class TieredOutput:
    """The three-tier compressed output."""

    topic: str
    overview: str
    clusters: list[Cluster] = field(default_factory=list)
    input_token_count: int = 0
    output_token_count: int = 0
    file_count: int = 0


@dataclass
class PipelineResult:
    """Full result of a compression run."""

    output: TieredOutput
    total_chunks: int = 0
    unique_chunks: int = 0
    duplicate_chunks: int = 0
    cluster_count: int = 0
