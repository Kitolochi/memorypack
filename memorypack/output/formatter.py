"""Render TieredOutput to markdown or multi-file format."""

from __future__ import annotations

import json

from memorypack.models import TieredOutput


def render_single(output: TieredOutput) -> str:
    """Render to a single markdown file."""
    ratio = (
        output.input_token_count / output.output_token_count
        if output.output_token_count > 0
        else 0
    )

    lines: list[str] = []
    lines.append(f"# Knowledge Base: {output.topic}")
    lines.append(
        f"> Compressed by memorypack | {output.file_count} files | "
        f"{output.input_token_count:,} → {output.output_token_count:,} tokens "
        f"({ratio:.1f}:1)"
    )
    lines.append("")

    # Tier 1: Overview
    lines.append("## Overview")
    lines.append(output.overview)
    lines.append("")

    # Tier 2: Topics with summaries
    lines.append("## Topics")
    for cluster in output.clusters:
        lines.append(f"### {cluster.label}")
        lines.append(cluster.summary)
        lines.append("")

    # Tier 3: Facts
    lines.append("## Facts")
    for cluster in output.clusters:
        if cluster.facts:
            lines.append(f"### {cluster.label}")
            for fact in cluster.facts:
                lines.append(f"- {fact}")
            lines.append("")

    return "\n".join(lines)


def render_multi(output: TieredOutput) -> dict[str, str]:
    """Render to multiple files: overview.md, facts.md, index.json."""
    ratio = (
        output.input_token_count / output.output_token_count
        if output.output_token_count > 0
        else 0
    )

    # overview.md
    overview_lines = [
        f"# Knowledge Base: {output.topic}",
        f"> Compressed by memorypack | {output.file_count} files | "
        f"{output.input_token_count:,} → {output.output_token_count:,} tokens "
        f"({ratio:.1f}:1)",
        "",
        "## Overview",
        output.overview,
        "",
    ]
    for cluster in output.clusters:
        overview_lines.append(f"### {cluster.label}")
        overview_lines.append(cluster.summary)
        overview_lines.append("")

    # facts.md
    facts_lines = [f"# Facts: {output.topic}", ""]
    for cluster in output.clusters:
        if cluster.facts:
            facts_lines.append(f"## {cluster.label}")
            for fact in cluster.facts:
                facts_lines.append(f"- {fact}")
            facts_lines.append("")

    # index.json
    index = {
        "topic": output.topic,
        "file_count": output.file_count,
        "input_tokens": output.input_token_count,
        "output_tokens": output.output_token_count,
        "compression_ratio": round(ratio, 1),
        "clusters": [
            {
                "label": c.label,
                "summary_length": len(c.summary),
                "fact_count": len(c.facts),
            }
            for c in output.clusters
        ],
    }

    return {
        "overview.md": "\n".join(overview_lines),
        "facts.md": "\n".join(facts_lines),
        "index.json": json.dumps(index, indent=2),
    }
