"""Spectral clustering with automatic k selection."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from memorypack.config import PipelineConfig
from memorypack.models import Chunk, Cluster


def _select_k(affinity: np.ndarray, min_k: int, max_k: int) -> int:
    """Select optimal number of clusters using silhouette score."""
    n = affinity.shape[0]
    max_k = min(max_k, n - 1)  # can't have more clusters than samples - 1
    if max_k <= min_k:
        return min_k

    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
                n_init=10,
            )
            labels = sc.fit_predict(affinity)
            # Need at least 2 unique labels for silhouette
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(affinity, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def _generate_label(chunks: list[Chunk]) -> str:
    """Generate a short topic label from chunk content.

    Looks for markdown headings first, then falls back to first words.
    """
    import re
    # Look for headings in chunk text
    for chunk in chunks:
        # Find markdown headings
        matches = re.findall(r"#+\s+(.+?)(?:\s+#|\.|$)", chunk.text)
        for label in matches:
            label = label.strip()
            # Only use if it looks like a heading (short, title-like)
            if 5 < len(label) < 50 and label[0].isupper():
                return label

    # Fallback: use first significant words from the longest chunk
    longest = max(chunks, key=lambda c: c.token_count)
    text = longest.text.lstrip("#").strip()
    # Take first sentence fragment as label
    for sep in (".", " — ", " - ", ","):
        if sep in text[:60]:
            text = text[:text.index(sep)]
            break
    else:
        words = text.split()[:5]
        text = " ".join(words)
    return text.strip().rstrip(".,;:")


def cluster_chunks(
    chunks: list[Chunk],
    similarity_matrix: np.ndarray,
    config: PipelineConfig,
) -> list[Cluster]:
    """Cluster chunks using spectral clustering."""
    n = len(chunks)

    if n <= 2:
        # Too few chunks to cluster meaningfully — put them all together
        cluster = Cluster(id=0, chunks=list(chunks), label=_generate_label(chunks))
        for c in chunks:
            c.cluster_id = 0
        return [cluster]

    # Convert similarity to affinity (ensure non-negative)
    affinity = np.clip(similarity_matrix, 0, 1)
    np.fill_diagonal(affinity, 1.0)

    # For silhouette with precomputed, we need distance matrix
    distance_matrix = 1 - affinity

    k = _select_k(distance_matrix, config.min_clusters, min(config.max_clusters, n - 1))

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(affinity)

    # Build Cluster objects
    cluster_map: dict[int, list[Chunk]] = {}
    for chunk, label in zip(chunks, labels):
        chunk.cluster_id = int(label)
        cluster_map.setdefault(int(label), []).append(chunk)

    clusters: list[Cluster] = []
    for cid in sorted(cluster_map):
        c_chunks = cluster_map[cid]
        clusters.append(
            Cluster(id=cid, chunks=c_chunks, label=_generate_label(c_chunks))
        )

    return clusters
