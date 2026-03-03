from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from ..schemas import ChunkRecord
from .model_loader import import_module_from_path


class Qwen3VLEmbeddingService:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        batch_size: int = 4,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.batch_size = max(1, int(batch_size))
        self._model = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        return torch.bfloat16

    def _build_kwargs(self) -> dict:
        kwargs = {}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            kwargs["torch_dtype"] = self._resolve_dtype()
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs

    def _load(self) -> None:
        if self._model is not None:
            return
        script = self.model_path / "scripts" / "qwen3_vl_embedding.py"
        module = import_module_from_path("qrag_qwen3_embed", script)
        self._model = module.Qwen3VLEmbedder(
            model_name_or_path=str(self.model_path),
            device=self.device,
            **self._build_kwargs(),
        )

    def _to_numpy(self, tensor_or_array) -> np.ndarray:
        if hasattr(tensor_or_array, "detach"):
            tensor = tensor_or_array.detach()
            # NumPy cannot directly consume bfloat16 tensors.
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            return tensor.cpu().numpy()
        return np.asarray(tensor_or_array)

    def embed_items(self, items: List[dict], normalize: bool = True) -> np.ndarray:
        if not items:
            return np.zeros((0, 1), dtype=np.float32)

        self._load()

        def run_batch(batch_items: List[dict]) -> np.ndarray:
            outputs = self._model.process(batch_items, normalize=normalize)
            arr = self._to_numpy(outputs)
            return arr.astype(np.float32)

        def run_with_retry(batch_items: List[dict]) -> np.ndarray:
            try:
                return run_batch(batch_items)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg or len(batch_items) == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mid = len(batch_items) // 2
                left = run_with_retry(batch_items[:mid])
                right = run_with_retry(batch_items[mid:])
                return np.concatenate([left, right], axis=0)

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            all_embeddings.append(run_with_retry(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def embed_query(self, query: str, instruction: str) -> np.ndarray:
        items = [{"text": query, "instruction": instruction}]
        return self.embed_items(items, normalize=True)[0]

    def embed_chunks(self, chunks: Iterable[ChunkRecord], instruction: str) -> np.ndarray:
        items = []
        for chunk in chunks:
            items.append({"text": chunk.text, "image": chunk.image_path, "instruction": instruction})
        if not items:
            return np.zeros((0, 1), dtype=np.float32)
        return self.embed_items(items, normalize=True)
