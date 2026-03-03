from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from ..schemas import PageRecord
from .model_loader import import_module_from_path


class Qwen3VLEmbeddingService:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
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
        self._load()
        outputs = self._model.process(items, normalize=normalize)
        arr = self._to_numpy(outputs)
        return arr.astype(np.float32)

    def embed_query(self, query: str, instruction: str) -> np.ndarray:
        items = [{"text": query, "instruction": instruction}]
        return self.embed_items(items, normalize=True)[0]

    def embed_pages(self, pages: Iterable[PageRecord], instruction: str) -> np.ndarray:
        items = []
        for page in pages:
            items.append({"text": page.text, "image": page.image_path, "instruction": instruction})
        if not items:
            return np.zeros((0, 1), dtype=np.float32)
        return self.embed_items(items, normalize=True)
