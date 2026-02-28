"""Click CLI entry point for memorypack."""

from __future__ import annotations

import click

from memorypack.config import PipelineConfig
from memorypack.output.formatter import render_single
from memorypack.output.stats import print_stats
from memorypack.output.writer import write_output
from memorypack.pipeline import run_pipeline
from memorypack.tokencount import estimate_tokens


@click.command()
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
def main(
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


if __name__ == "__main__":
    main()
