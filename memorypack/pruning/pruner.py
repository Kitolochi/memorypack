"""Execute pruning: remove low-importance clusters, merge near-duplicates."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from memorypack.config import PruneConfig
from memorypack.models import Cluster, PruneResult, TieredOutput
from memorypack.pruning.analyzer import find_near_duplicates, score_clusters
from memorypack.tokencount import estimate_tokens


def _merge_clusters(keep: Cluster, remove: Cluster) -> Cluster:
    """Merge two clusters, combining facts and keeping the longer summary."""
    merged_facts = list(keep.facts)
    existing = set(keep.facts)
    for fact in remove.facts:
        if fact not in existing:
            merged_facts.append(fact)

    summary = keep.summary if len(keep.summary) >= len(remove.summary) else remove.summary
    label = keep.label

    return Cluster(
        id=keep.id,
        label=label,
        summary=summary,
        facts=merged_facts,
        chunks=keep.chunks + remove.chunks,
    )


def _estimate_output_tokens(output: TieredOutput) -> int:
    """Estimate token count for the pruned output."""
    parts = [output.overview]
    for c in output.clusters:
        parts.append(c.summary)
        parts.extend(c.facts)
    return estimate_tokens("\n".join(parts))


def prune(
    output: TieredOutput,
    config: PruneConfig,
    encoder: SentenceTransformer | None = None,
) -> PruneResult:
    """Prune a TieredOutput to fit within token budget.

    Steps:
      1. Score all clusters by importance
      2. If merging enabled, detect and merge near-duplicate clusters
      3. Remove lowest-importance clusters until under max_tokens budget
    """
    original_count = len(output.clusters)
    removed_labels: list[str] = []
    merged_pairs: list[tuple[str, str]] = []

    scores = score_clusters(output)

    # Step 1: Merge near-duplicates if enabled and encoder is available
    if config.merge_duplicates and encoder is not None:
        duplicates = find_near_duplicates(output, encoder, config.similarity_threshold)

        # Sort by similarity descending — merge most similar first
        duplicates.sort(key=lambda x: x[2], reverse=True)

        cluster_map = {c.id: c for c in output.clusters}
        removed_ids: set[int] = set()

        for id_a, id_b, _sim in duplicates:
            if id_a in removed_ids or id_b in removed_ids:
                continue
            # Keep the higher-importance cluster
            if scores.get(id_a, 0) >= scores.get(id_b, 0):
                keep_id, remove_id = id_a, id_b
            else:
                keep_id, remove_id = id_b, id_a

            keep = cluster_map[keep_id]
            remove = cluster_map[remove_id]
            cluster_map[keep_id] = _merge_clusters(keep, remove)
            removed_ids.add(remove_id)
            merged_pairs.append((keep.label, remove.label))

        output.clusters = [
            cluster_map[c.id]
            for c in output.clusters
            if c.id not in removed_ids
        ]

    # Step 2: Remove low-importance clusters if min_importance is set
    if config.min_importance > 0:
        before = [c.label for c in output.clusters]
        output.clusters = [
            c for c in output.clusters
            if scores.get(c.id, 0) >= config.min_importance
        ]
        after = {c.label for c in output.clusters}
        removed_labels.extend(l for l in before if l not in after)

    # Step 3: Remove lowest-importance clusters to meet token budget
    if config.max_tokens > 0:
        while _estimate_output_tokens(output) > config.max_tokens and len(output.clusters) > 1:
            # Find and remove the least important remaining cluster
            worst = min(output.clusters, key=lambda c: scores.get(c.id, 0))
            removed_labels.append(worst.label)
            output.clusters = [c for c in output.clusters if c.id != worst.id]

    final_tokens = _estimate_output_tokens(output)

    return PruneResult(
        output=output,
        original_cluster_count=original_count,
        pruned_cluster_count=len(output.clusters),
        removed_labels=removed_labels,
        merged_pairs=merged_pairs,
        output_token_count=final_tokens,
    )
