from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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

    def describe_visual_evidence(
        self,
        query: str,
        evidence: List[EvidenceItem],
        image_max_pixels: int,
        max_items: int = 3,
    ) -> Dict[str, str]:
        """Generate concise descriptions for visual evidence chunks keyed by chunk ref."""
        figure_items = [
            item
            for item in evidence
            if item.chunk_type.lower() in {"figure", "image", "table", "chart"}
        ][: max(0, max_items)]
        if not figure_items:
            return {}

        desc_cfg = GenerationConfig(max_new_tokens=220, temperature=0.1, force_table=False)
        descriptions: Dict[str, str] = {}
        for item in figure_items:
            ref_id = make_chunk_ref(item.page_num, item.order)
            prompt = (
                "You are analyzing one scientific evidence image from a PDF.\n"
                "Task:\n"
                "1) Describe what is visually present in the image.\n"
                "2) Explain the most question-relevant signal from this image.\n"
                "Constraints:\n"
                "- Use concise Chinese.\n"
                "- Do not invent details not visible in the image.\n"
                "- Keep within at most ten sentences.\n\n"
                f"Question: {query}\n"
                f"Evidence ref: [{ref_id}]\n"
                f"Known snippet: {short_snippet(item.snippet, 200)}\n"
                "Return plain text only."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{item.image_path}",
                            "max_pixels": image_max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            descriptions[ref_id] = self._generate_from_messages(messages, desc_cfg)
        return descriptions

    def caption_visual_chunks(
        self,
        visual_items: List[dict],
        image_max_pixels: int,
        max_items: int = 24,
    ) -> Dict[str, str]:
        """Generate neutral captions for visual chunks during indexing."""
        if not visual_items:
            return {}

        cap_cfg = GenerationConfig(max_new_tokens=120, temperature=0.1, force_table=False)
        captions: Dict[str, str] = {}
        for item in visual_items[: max(0, max_items)]:
            ref_id = str(item.get("ref_id", "")).strip()
            image_path = str(item.get("image_path", "")).strip()
            hint = short_snippet(str(item.get("hint", "")), 180)
            if not ref_id or not image_path:
                continue
            prompt = (
                "You are captioning a scientific PDF image chunk for retrieval indexing.\n"
                "Write a concise Chinese description that captures the main visual content.\n"
                "If there are visible chart/table/axes/title cues, mention them briefly.\n"
                "Do not hallucinate precise values.\n"
                "Keep within one sentence.\n"
                f"Chunk ref: [{ref_id}]\n"
                f"Optional text hint: {hint}\n"
                "Return plain text only."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                            "max_pixels": image_max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            captions[ref_id] = self._generate_from_messages(messages, cap_cfg)
        return captions
