"""Pipeline orchestrator — wires all stages together."""

from __future__ import annotations

import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_length.*input_length.*")
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from memorypack.clustering.cluster import cluster_chunks
from memorypack.clustering.dedup import deduplicate
from memorypack.config import PipelineConfig
from memorypack.embedding.encoder import (
    build_similarity_matrix,
    encode_chunks,
    load_encoder,
)
from memorypack.models import PipelineResult, TieredOutput
from memorypack.parsing.chunker import chunk_text
from memorypack.parsing.cleaner import clean_markdown
from memorypack.parsing.reader import read_files
from memorypack.summarization.fact_extractor import extract_facts
from memorypack.summarization.summarizer import (
    generate_overview,
    load_summarizer,
    summarize_cluster,
)
from memorypack.tokencount import estimate_tokens


def run_pipeline(input_paths: list[str], config: PipelineConfig) -> PipelineResult:
    """Execute the full compression pipeline."""
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. Read files
        task = progress.add_task("Reading markdown files...", total=None)
        source_files = read_files(input_paths)
        if not source_files:
            console.print("[red]No markdown files found.[/red]")
            raise SystemExit(1)
        progress.update(task, description=f"Read {len(source_files)} files")
        progress.remove_task(task)

        # Calculate input tokens
        input_token_count = sum(estimate_tokens(sf.body) for sf in source_files)

        # 2. Clean
        task = progress.add_task("Cleaning markdown...", total=None)
        for sf in source_files:
            sf.body = clean_markdown(sf.body)
        progress.remove_task(task)

        # 3. Chunk
        task = progress.add_task("Chunking into semantic blocks...", total=None)
        all_chunks = []
        chunk_id = 0
        for sf in source_files:
            chunks = chunk_text(sf.body, sf.path, config.chunk_size, chunk_id)
            all_chunks.extend(chunks)
            chunk_id += len(chunks)
        total_chunks = len(all_chunks)
        progress.update(task, description=f"Created {total_chunks} chunks")
        progress.remove_task(task)

        if not all_chunks:
            console.print("[red]No content to process.[/red]")
            raise SystemExit(1)

        # 4. Encode
        task = progress.add_task("Computing embeddings...", total=None)
        encoder = load_encoder(config)
        embeddings = encode_chunks(all_chunks, encoder)
        similarity_matrix = build_similarity_matrix(embeddings)
        progress.remove_task(task)

        # 5. Deduplicate
        task = progress.add_task("Deduplicating...", total=None)
        unique_chunks = deduplicate(all_chunks, similarity_matrix, config.dedup_threshold)
        duplicate_count = total_chunks - len(unique_chunks)
        progress.update(
            task, description=f"Removed {duplicate_count} duplicates"
        )
        progress.remove_task(task)

        # Rebuild embeddings/similarity for unique chunks only
        if len(unique_chunks) < total_chunks:
            unique_indices = [c.id for c in unique_chunks]
            # Re-index: build a new similarity matrix from the original
            idx_map = {orig_id: i for i, orig_id in enumerate(range(len(all_chunks)))}
            sel = [idx_map[c.id] for c in unique_chunks if c.id in idx_map]
            if sel:
                unique_embeddings = embeddings[sel]
                unique_sim = build_similarity_matrix(unique_embeddings)
            else:
                unique_embeddings = embeddings
                unique_sim = similarity_matrix
        else:
            unique_embeddings = embeddings
            unique_sim = similarity_matrix

        # 6. Cluster
        task = progress.add_task("Clustering topics...", total=None)
        clusters = cluster_chunks(unique_chunks, unique_sim, config)
        progress.update(task, description=f"Found {len(clusters)} topics")
        progress.remove_task(task)

        # 7. Summarize
        task = progress.add_task("Loading summarization model...", total=None)
        summarizer = load_summarizer(config)
        progress.remove_task(task)

        task = progress.add_task("Summarizing clusters...", total=None)
        for cluster in clusters:
            cluster.summary = summarize_cluster(cluster, summarizer, config)
            progress.update(task, description=f"Summarized: {cluster.label}")
        progress.remove_task(task)

        # 8. Extract facts
        task = progress.add_task("Extracting facts...", total=None)
        for cluster in clusters:
            cluster.facts = extract_facts(cluster)
        progress.remove_task(task)

        # 9. Generate overview
        task = progress.add_task("Generating overview...", total=None)
        overview = generate_overview(clusters, summarizer, config)
        progress.remove_task(task)

    # Assemble output
    tiered = TieredOutput(
        topic=config.topic,
        overview=overview,
        clusters=clusters,
        input_token_count=input_token_count,
        output_token_count=0,  # will be calculated after formatting
        file_count=len(source_files),
    )

    return PipelineResult(
        output=tiered,
        total_chunks=total_chunks,
        unique_chunks=len(unique_chunks),
        duplicate_chunks=duplicate_count,
        cluster_count=len(clusters),
    )
