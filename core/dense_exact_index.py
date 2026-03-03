from __future__ import annotations

from typing import List

import numpy as np

from ..schemas import RetrievalCandidate
from .vector_index import VectorIndex


class DenseExactIndex(VectorIndex):
    """Exact cosine similarity search on in-memory dense vectors."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._metadata: List[dict] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        arr = vectors.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    def build(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings rows must match metadata size")
        self._embeddings = self._normalize(embeddings)
        self._metadata = metadata

    def is_ready(self) -> bool:
        return self._embeddings is not None and bool(self._metadata)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalCandidate]:
        if self._embeddings is None:
            raise RuntimeError("Index is not built")

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 2:
            if query.shape[0] != 1:
                raise ValueError("Only single query search is supported")
            query = query[0]
        query = query / max(np.linalg.norm(query), 1e-12)

        scores = self._embeddings @ query
        k = max(1, min(top_k, scores.shape[0]))
        idxs = np.argpartition(scores, -k)[-k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

        results: List[RetrievalCandidate] = []
        for i in idxs.tolist():
            meta = self._metadata[i]
            results.append(
                RetrievalCandidate(
                    page_num=int(meta["page_num"]),
                    text=str(meta["text"]),
                    image_path=str(meta["image_path"]),
                    score=float(scores[i]),
                )
            )
        return results
