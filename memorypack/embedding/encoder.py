"""MiniLM embedding and cosine similarity matrix."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from memorypack.config import PipelineConfig
from memorypack.models import Chunk


def load_encoder(config: PipelineConfig) -> SentenceTransformer:
    """Load the sentence-transformer model."""
    return SentenceTransformer(config.embedding_model, device=config.device)


def encode_chunks(
    chunks: list[Chunk], model: SentenceTransformer
) -> np.ndarray:
    """Compute embeddings for all chunks, returning (N, 384) array."""
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb.tolist()
    return embeddings


def build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    return cosine_similarity(embeddings)
