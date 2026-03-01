"""Parse a knowledge_base.md file back into a TieredOutput structure."""

from __future__ import annotations

import re
from pathlib import Path

from memorypack.models import Cluster, TieredOutput


def parse_knowledge_base(path: str) -> TieredOutput:
    """Parse a memorypack knowledge_base.md file into TieredOutput.

    Expected structure:
        # Knowledge Base: <topic>
        > Compressed by memorypack | N files | X -> Y tokens (R:1)

        ## Overview
        <overview text>

        ## Topics
        ### <cluster label>
        <summary text>

        ## Facts
        ### <cluster label>
        - fact 1
        - fact 2
    """
    text = Path(path).read_text(encoding="utf-8")

    # Extract topic from header
    topic_match = re.search(r"^# Knowledge Base:\s*(.+)$", text, re.MULTILINE)
    topic = topic_match.group(1).strip() if topic_match else "Knowledge Base"

    # Extract token counts from the blockquote
    meta_match = re.search(
        r"(\d[\d,]*)\s*→\s*(\d[\d,]*)\s*tokens", text
    )
    input_tokens = int(meta_match.group(1).replace(",", "")) if meta_match else 0
    output_tokens = int(meta_match.group(2).replace(",", "")) if meta_match else 0

    file_count_match = re.search(r"(\d+)\s*files", text)
    file_count = int(file_count_match.group(1)) if file_count_match else 0

    # Split into major sections
    overview = ""
    overview_match = re.search(
        r"## Overview\n(.*?)(?=\n## )", text, re.DOTALL
    )
    if overview_match:
        overview = overview_match.group(1).strip()

    # Parse topic summaries
    clusters: dict[str, Cluster] = {}
    topics_match = re.search(
        r"## Topics\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    )
    if topics_match:
        topics_text = topics_match.group(1)
        topic_blocks = re.split(r"### ", topics_text)
        for i, block in enumerate(topic_blocks):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n", 1)
            label = lines[0].strip()
            summary = lines[1].strip() if len(lines) > 1 else ""
            clusters[label] = Cluster(
                id=i, label=label, summary=summary, facts=[]
            )

    # Parse facts
    facts_match = re.search(
        r"## Facts\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    )
    if facts_match:
        facts_text = facts_match.group(1)
        fact_blocks = re.split(r"### ", facts_text)
        for block in fact_blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n", 1)
            label = lines[0].strip()
            if label in clusters and len(lines) > 1:
                facts = [
                    line.lstrip("- ").strip()
                    for line in lines[1].strip().split("\n")
                    if line.strip().startswith("- ")
                ]
                clusters[label].facts = facts

    return TieredOutput(
        topic=topic,
        overview=overview,
        clusters=list(clusters.values()),
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        file_count=file_count,
    )
