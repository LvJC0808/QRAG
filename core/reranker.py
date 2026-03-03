from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from ..schemas import RetrievalCandidate
from .model_loader import import_module_from_path


class Qwen3VLRerankerService:
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
        script = self.model_path / "scripts" / "qwen3_vl_reranker.py"
        module = import_module_from_path("qrag_qwen3_rerank", script)
        self._model = module.Qwen3VLReranker(
            model_name_or_path=str(self.model_path),
            device=self.device,
            **self._build_kwargs(),
        )

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        instruction: str,
        top_n: int,
    ) -> List[tuple[RetrievalCandidate, float]]:
        if not candidates:
            return []

        self._load()
        documents = [{"text": cand.text, "image": cand.image_path} for cand in candidates]
        scores = self._model.process(
            {
                "instruction": instruction,
                "query": {"text": query},
                "documents": documents,
            }
        )

        merged = list(zip(candidates, [float(s) for s in scores]))
        merged.sort(key=lambda item: item[1], reverse=True)
        return merged[: max(1, min(top_n, len(merged)))]
