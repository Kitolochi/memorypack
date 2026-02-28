"""Discover and read markdown files."""

from __future__ import annotations

from pathlib import Path

import frontmatter

from memorypack.models import SourceFile


def discover_files(paths: list[str]) -> list[Path]:
    """Resolve a list of paths/globs into .md file paths."""
    results: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".md":
            results.append(path)
        elif path.is_dir():
            results.extend(sorted(path.rglob("*.md")))
    return results


def read_file(path: Path) -> SourceFile:
    """Read a markdown file, stripping frontmatter."""
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    return SourceFile(
        path=str(path),
        raw_content=raw,
        metadata=dict(post.metadata),
        body=post.content,
    )


def read_files(paths: list[str]) -> list[SourceFile]:
    """Discover and read all markdown files from the given paths."""
    file_paths = discover_files(paths)
    return [read_file(p) for p in file_paths]
