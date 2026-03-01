"""Click CLI entry point for memorypack."""

from __future__ import annotations

import click

from memorypack.config import PipelineConfig
from memorypack.output.formatter import render_single
from memorypack.output.stats import print_stats
from memorypack.output.writer import write_output
from memorypack.pipeline import run_pipeline
from memorypack.tokencount import estimate_tokens


@click.group()
def cli() -> None:
    """memorypack — compress markdown knowledge bases for LLMs."""


@cli.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path())
@click.option(
    "-o", "--output", "output_dir", default="compressed", help="Output directory."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["single", "multi"]),
    default="single",
    help="Output format: single .md file or multi-file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Device for model inference.",
)
@click.option(
    "--compression-target",
    type=float,
    default=6.0,
    help="Target compression ratio.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Target tokens per chunk.",
)
@click.option(
    "--topic",
    type=str,
    default="Knowledge Base",
    help="Topic name for the output header.",
)
def compress(
    inputs: tuple[str, ...],
    output_dir: str,
    fmt: str,
    device: str,
    compression_target: float,
    chunk_size: int,
    topic: str,
) -> None:
    """Compress markdown files into context-efficient format for LLMs."""
    config = PipelineConfig(
        chunk_size=chunk_size,
        device=device,
        compression_target=compression_target,
        output_format=fmt,
        topic=topic,
    )

    result = run_pipeline(list(inputs), config)

    # Calculate output token count from the rendered content
    rendered = render_single(result.output)
    result.output.output_token_count = estimate_tokens(rendered)

    # Write to disk
    written = write_output(result.output, output_dir, fmt)

    # Print stats
    print_stats(result)

    click.echo()
    for f in written:
        click.echo(f"  Written: {f}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--max-tokens",
    type=int,
    default=0,
    help="Maximum output token budget (0 = no limit).",
)
@click.option(
    "--min-importance",
    type=float,
    default=0.0,
    help="Drop clusters below this importance score [0-1].",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.80,
    help="Cosine threshold for near-duplicate detection.",
)
@click.option(
    "--no-merge",
    is_flag=True,
    default=False,
    help="Disable merging of near-duplicate clusters.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print prune plan without writing output.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Device for embedding inference.",
)
@click.option(
    "-o", "--output", "output_path", default=None, help="Output file path.",
)
def prune(
    input_path: str,
    max_tokens: int,
    min_importance: float,
    similarity_threshold: float,
    no_merge: bool,
    dry_run: bool,
    device: str,
    output_path: str | None,
) -> None:
    """Prune a knowledge_base.md to reduce token count."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from memorypack.config import PruneConfig
    from memorypack.pruning.parser import parse_knowledge_base
    from memorypack.pruning.pruner import prune as run_prune

    console = Console()

    # Parse existing knowledge base
    output = parse_knowledge_base(input_path)
    console.print(f"Parsed {len(output.clusters)} clusters from {input_path}")

    config = PruneConfig(
        max_tokens=max_tokens,
        min_importance=min_importance,
        similarity_threshold=similarity_threshold,
        merge_duplicates=not no_merge,
        dry_run=dry_run,
        device=device,
    )

    # Load encoder only if merging is enabled
    encoder = None
    if config.merge_duplicates and len(output.clusters) >= 2:
        from memorypack.embedding.encoder import load_encoder
        from memorypack.config import PipelineConfig

        pipe_config = PipelineConfig(device=device)
        encoder = load_encoder(pipe_config)

    result = run_prune(output, config, encoder)

    # Display results
    table = Table(title="Prune Results", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Original clusters", str(result.original_cluster_count))
    table.add_row("Pruned clusters", str(result.pruned_cluster_count))
    table.add_row("Output tokens", f"{result.output_token_count:,}")

    if result.merged_pairs:
        for keep, removed in result.merged_pairs:
            table.add_row("Merged", f"{removed} → {keep}")

    if result.removed_labels:
        table.add_row("Removed", ", ".join(result.removed_labels))

    console.print()
    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run — no files written.[/yellow]")
        return

    # Write output
    dest = output_path or input_path
    rendered = render_single(result.output)
    Path(dest).write_text(rendered, encoding="utf-8")
    console.print(f"\n  Written: {dest}")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--interval",
    type=int,
    default=60,
    help="Polling interval in seconds.",
)
@click.option(
    "--token-budget",
    type=int,
    default=0,
    help="Auto-prune if output exceeds this token count (0 = no limit).",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Device for model inference.",
)
@click.option(
    "--topic",
    type=str,
    default="Knowledge Base",
    help="Topic name for the output header.",
)
@click.option(
    "-o", "--output", "output_dir", default="compressed", help="Output directory.",
)
def watch(
    input_dir: str,
    interval: int,
    token_budget: int,
    device: str,
    topic: str,
    output_dir: str,
) -> None:
    """Watch a directory for .md changes and auto-recompress."""
    from memorypack.watcher import MemorypackWatcher

    watcher = MemorypackWatcher(
        input_dir=input_dir,
        output_dir=output_dir,
        interval=interval,
        token_budget=token_budget,
        device=device,
        topic=topic,
    )
    watcher.run()


if __name__ == "__main__":
    cli()
