"""Analyze clusters for importance scoring and near-duplicate detection."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from memorypack.models import Cluster, TieredOutput


def score_clusters(output: TieredOutput) -> dict[int, float]:
    """Score each cluster by importance.

    Importance = weighted combination of:
      - fact_count (weight 0.6) — more facts = more specific knowledge
      - summary_length in words (weight 0.4) — longer summaries = richer content

    Returns dict mapping cluster.id -> normalized importance score [0, 1].
    """
    if not output.clusters:
        return {}

    raw_scores: dict[int, float] = {}
    max_facts = max(len(c.facts) for c in output.clusters) or 1
    max_words = max(len(c.summary.split()) for c in output.clusters) or 1

    for cluster in output.clusters:
        fact_score = len(cluster.facts) / max_facts
        summary_score = len(cluster.summary.split()) / max_words
        raw_scores[cluster.id] = 0.6 * fact_score + 0.4 * summary_score

    return raw_scores


def find_near_duplicates(
    output: TieredOutput,
    encoder: SentenceTransformer,
    threshold: float = 0.80,
) -> list[tuple[int, int, float]]:
    """Detect near-duplicate cluster pairs via summary embedding similarity.

    Returns list of (cluster_id_a, cluster_id_b, similarity) where
    similarity >= threshold. The pair with lower importance should be merged
    into the higher one.
    """
    if len(output.clusters) < 2:
        return []

    summaries = [c.summary for c in output.clusters]
    embeddings = encoder.encode(summaries, show_progress_bar=False, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)

    duplicates: list[tuple[int, int, float]] = []
    for i in range(len(output.clusters)):
        for j in range(i + 1, len(output.clusters)):
            if sim_matrix[i][j] >= threshold:
                duplicates.append(
                    (output.clusters[i].id, output.clusters[j].id, float(sim_matrix[i][j]))
                )

    return duplicates
