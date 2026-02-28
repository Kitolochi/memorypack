"""Compression statistics display."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from memorypack.models import PipelineResult


def print_stats(result: PipelineResult) -> None:
    """Print a rich table of compression statistics."""
    console = Console()
    output = result.output

    ratio = (
        output.input_token_count / output.output_token_count
        if output.output_token_count > 0
        else 0
    )

    table = Table(title="Compression Statistics", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Input files", str(output.file_count))
    table.add_row("Input tokens", f"{output.input_token_count:,}")
    table.add_row("Output tokens", f"{output.output_token_count:,}")
    table.add_row("Compression ratio", f"{ratio:.1f}:1")
    table.add_row("Total chunks", str(result.total_chunks))
    table.add_row("Unique chunks", str(result.unique_chunks))
    table.add_row("Duplicates removed", str(result.duplicate_chunks))
    table.add_row("Clusters", str(result.cluster_count))
    table.add_row("Topics", ", ".join(c.label for c in output.clusters))

    console.print()
    console.print(table)
