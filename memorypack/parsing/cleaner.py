"""Normalize markdown content for processing."""

from __future__ import annotations

import re


def clean_markdown(text: str) -> str:
    """Normalize markdown while preserving semantic structure."""
    # Collapse multiple blank lines to single
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace within lines (but preserve newlines)
    lines = []
    for line in text.split("\n"):
        # Strip trailing whitespace
        line = line.rstrip()
        # Collapse multiple spaces (but not leading indentation)
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        stripped = re.sub(r"  +", " ", stripped)
        lines.append(indent + stripped)
    text = "\n".join(lines)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Normalize link references to inline form isn't needed — just strip ref defs
    # Remove reference-style link definitions
    text = re.sub(r"^\[.+?\]:\s+\S+.*$", "", text, flags=re.MULTILINE)

    # Remove images (keep alt text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Simplify horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "---", text, flags=re.MULTILINE)

    # Strip trailing whitespace and collapse blank lines again
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
