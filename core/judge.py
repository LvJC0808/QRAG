from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..schemas import DimensionScores, EvidenceItem, JudgeResult
from .utils import make_chunk_ref, short_snippet


class Qwen3VLJudgeService:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        judge_template: str = "",
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.judge_template = judge_template
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._model = None
        self._processor = None

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        return torch.bfloat16

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        kwargs = {"trust_remote_code": True}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            kwargs["torch_dtype"] = self._resolve_dtype()
            kwargs["attn_implementation"] = self.attn_implementation
            kwargs["device_map"] = {"": self.device}
        else:
            kwargs["device_map"] = "cpu"

        self._processor = AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), **kwargs)
        self._model.eval()

    def _text_generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        self._load()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    @staticmethod
    def parse_result(raw_text: str) -> JudgeResult:
        text = raw_text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return JudgeResult(
                overall_score=0.0,
                dimension_scores=DimensionScores(),
                major_issues=["Judge output was not valid JSON."],
                actionable_feedback=["Regenerate answer with stricter citation and factual constraints."],
                verdict="revise",
                raw_text=raw_text,
            )

        dims = payload.get("dimension_scores", {}) if isinstance(payload, dict) else {}
        result = JudgeResult(
            overall_score=float(payload.get("overall_score", 0.0)),
            dimension_scores=DimensionScores(
                relevance=float(dims.get("relevance", 0.0)),
                groundedness=float(dims.get("groundedness", 0.0)),
                completeness=float(dims.get("completeness", 0.0)),
                numeric_consistency=float(dims.get("numeric_consistency", 0.0)),
                citation_validity=float(dims.get("citation_validity", 0.0)),
            ),
            major_issues=[str(item) for item in payload.get("major_issues", [])],
            actionable_feedback=[str(item) for item in payload.get("actionable_feedback", [])],
            verdict=str(payload.get("verdict", "revise")).lower(),
            raw_text=raw_text,
        )
        if result.verdict not in {"accept", "revise"}:
            result.verdict = "revise"
        return result

    def evaluate(self, query: str, answer: str, evidence_items: List[EvidenceItem]) -> JudgeResult:
        evidence_lines = [
            f"[{make_chunk_ref(item.page_num, item.order)}] [p{item.page_num}] "
            f"{short_snippet(item.snippet, 220)}"
            for item in evidence_items
        ]
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "No evidence provided"

        prompt = (
            f"{self.judge_template}\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence snippets:\n{evidence_block}\n\n"
            f"Answer to evaluate:\n{answer}\n"
        )
        raw = self._text_generate(prompt)
        return self.parse_result(raw)
