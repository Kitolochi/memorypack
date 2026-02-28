"""Write output files to disk."""

from __future__ import annotations

from pathlib import Path

from memorypack.models import TieredOutput
from memorypack.output.formatter import render_multi, render_single


def write_output(output: TieredOutput, output_dir: str, fmt: str = "single") -> list[str]:
    """Write compressed output to disk. Returns list of written file paths."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    if fmt == "multi":
        files = render_multi(output)
        for filename, content in files.items():
            fpath = out_path / filename
            fpath.write_text(content, encoding="utf-8")
            written.append(str(fpath))
    else:
        content = render_single(output)
        fpath = out_path / "knowledge_base.md"
        fpath.write_text(content, encoding="utf-8")
        written.append(str(fpath))

    return written
