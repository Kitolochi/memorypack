"""File watcher — polls for .md changes, re-runs compress, auto-prunes."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from memorypack.config import PipelineConfig, PruneConfig
from memorypack.output.formatter import render_single
from memorypack.output.writer import write_output
from memorypack.pipeline import run_pipeline
from memorypack.tokencount import estimate_tokens


class MemorypackWatcher:
    """Polls a directory for .md file changes and re-runs compression."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        interval: int = 60,
        token_budget: int = 0,
        device: str = "cpu",
        topic: str = "Knowledge Base",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = output_dir
        self.interval = interval
        self.token_budget = token_budget
        self.device = device
        self.topic = topic
        self.console = Console()
        self._last_mtimes: dict[str, float] = {}

    def _scan_mtimes(self) -> dict[str, float]:
        """Get mtime for all .md files in input directory."""
        mtimes: dict[str, float] = {}
        for md_file in self.input_dir.rglob("*.md"):
            mtimes[str(md_file)] = md_file.stat().st_mtime
        return mtimes

    def _has_changes(self) -> bool:
        """Check if any .md files have been added, removed, or modified."""
        current = self._scan_mtimes()
        if current != self._last_mtimes:
            self._last_mtimes = current
            return True
        return False

    def _run_compress(self) -> None:
        """Run compression pipeline on all .md files in input dir."""
        md_files = sorted(str(f) for f in self.input_dir.rglob("*.md"))
        if not md_files:
            self.console.print("[yellow]No .md files found.[/yellow]")
            return

        config = PipelineConfig(
            device=self.device,
            topic=self.topic,
        )

        try:
            result = run_pipeline(md_files, config)

            rendered = render_single(result.output)
            result.output.output_token_count = estimate_tokens(rendered)

            written = write_output(result.output, self.output_dir, "single")

            self.console.print(
                f"[green]Compressed {len(md_files)} files → "
                f"{result.output.output_token_count:,} tokens[/green]"
            )

            # Auto-prune if over budget
            if self.token_budget > 0 and result.output.output_token_count > self.token_budget:
                self._auto_prune(written[0])

        except Exception as e:
            self.console.print(f"[red]Compression failed: {e}[/red]")

    def _auto_prune(self, output_path: str) -> None:
        """Auto-prune if output exceeds token budget."""
        from memorypack.pruning.parser import parse_knowledge_base
        from memorypack.pruning.pruner import prune as run_prune

        self.console.print(
            f"[yellow]Over budget ({self.token_budget:,} tokens), auto-pruning...[/yellow]"
        )

        output = parse_knowledge_base(output_path)
        prune_config = PruneConfig(
            max_tokens=self.token_budget,
            device=self.device,
        )

        result = run_prune(output, prune_config)
        rendered = render_single(result.output)
        Path(output_path).write_text(rendered, encoding="utf-8")

        self.console.print(
            f"[green]Pruned to {result.pruned_cluster_count} clusters, "
            f"~{result.output_token_count:,} tokens[/green]"
        )

    def run(self) -> None:
        """Start the polling loop. Runs until interrupted."""
        self.console.print(
            f"Watching [bold]{self.input_dir}[/bold] every {self.interval}s..."
        )

        # Initial run
        self._last_mtimes = self._scan_mtimes()
        self._run_compress()

        try:
            while True:
                time.sleep(self.interval)
                if self._has_changes():
                    self.console.print("\n[cyan]Changes detected, recompressing...[/cyan]")
                    self._run_compress()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Watch stopped.[/yellow]")
