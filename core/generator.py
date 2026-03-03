from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..schemas import EvidenceItem, GenerationConfig
from .utils import make_chunk_ref, short_snippet


class Qwen3VLGeneratorService:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        prompt_template: str = "",
        revision_template: str = "",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.prompt_template = prompt_template
        self.revision_template = revision_template
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

    def _generate_from_messages(self, messages: List[dict], config: GenerationConfig) -> str:
        self._load()

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
            )

        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def _build_evidence_payload(self, evidence: List[EvidenceItem], max_pixels: int) -> List[dict]:
        payload: List[dict] = []
        for item in evidence:
            ref_id = make_chunk_ref(item.page_num, item.order)
            payload.append(
                {
                    "type": "image",
                    "image": f"file://{item.image_path}",
                    "max_pixels": max_pixels,
                }
            )
            payload.append(
                {
                    "type": "text",
                    "text": (
                        f"[{ref_id}] page={item.page_num} order={item.order} type={item.chunk_type} "
                        f"snippet: {short_snippet(item.snippet, 280)}"
                    ),
                }
            )
        return payload

    def generate_answer(
        self,
        query: str,
        evidence: List[EvidenceItem],
        config: GenerationConfig,
        image_max_pixels: int,
    ) -> str:
        evidence_payload = self._build_evidence_payload(evidence, image_max_pixels)
        text_prompt = (
            f"{self.prompt_template}\n\n"
            f"User question: {query}\n"
            "Use only provided evidence, cite with [pX-cY], and keep the required section headers."
        )

        messages = [
            {
                "role": "user",
                "content": evidence_payload + [{"type": "text", "text": text_prompt}],
            }
        ]
        return self._generate_from_messages(messages, config)

    def revise_answer(
        self,
        query: str,
        draft_answer: str,
        judge_feedback: List[str],
        evidence: List[EvidenceItem],
        config: GenerationConfig,
        image_max_pixels: int,
    ) -> str:
        evidence_payload = self._build_evidence_payload(evidence, image_max_pixels)
        feedback_block = "\n".join(f"- {item}" for item in judge_feedback) or "- No explicit feedback"
        prompt = (
            f"{self.revision_template}\n\n"
            f"Question: {query}\n"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Judge feedback:\n{feedback_block}\n"
            "Return only the revised final answer."
        )
        messages = [{"role": "user", "content": evidence_payload + [{"type": "text", "text": prompt}]}]
        return self._generate_from_messages(messages, config)
