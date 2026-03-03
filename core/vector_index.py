from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..schemas import RetrievalCandidate


class VectorIndex(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalCandidate]:
        raise NotImplementedError
