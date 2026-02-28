"""Union-find based deduplication of near-identical chunks."""

from __future__ import annotations

import numpy as np

from memorypack.models import Chunk


class UnionFind:
    """Simple union-find / disjoint-set structure."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def deduplicate(
    chunks: list[Chunk], similarity_matrix: np.ndarray, threshold: float = 0.92
) -> list[Chunk]:
    """Mark near-duplicate chunks. Returns only unique chunks.

    Uses union-find to group duplicates, keeps the one with most tokens.
    """
    n = len(chunks)
    uf = UnionFind(n)

    # Find pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                uf.union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    # Keep the longest chunk in each group, mark others as duplicates
    unique: list[Chunk] = []
    for indices in groups.values():
        # Pick the chunk with the most tokens as representative
        best = max(indices, key=lambda i: chunks[i].token_count)
        for i in indices:
            if i != best:
                chunks[i].is_duplicate = True
        unique.append(chunks[best])

    return unique
