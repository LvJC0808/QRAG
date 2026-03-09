from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

import fitz

from ..schemas import ChunkRecord
from .utils import make_doc_id


@dataclass(slots=True)
class _TextBlock:
    text: str
    bbox: tuple[float, float, float, float]


class PDFIngestor:
    def __init__(
        self,
        cache_root: Path,
        dpi: int = 600,
        text_chunk_chars: int = 1200,
        text_chunk_tokens: int = 220,
        text_overlap_tokens: int = 40,
        caption_window: float = 90.0,
    ) -> None:
        self.cache_root = cache_root
        self.dpi = dpi
        self.text_chunk_chars = text_chunk_chars
        self.text_chunk_tokens = max(64, int(text_chunk_tokens))
        self.text_overlap_tokens = max(0, int(text_overlap_tokens))
        self.caption_window = caption_window

    @staticmethod
    def _block_text(block: dict) -> str:
        lines = block.get("lines", [])
        spans: list[str] = []
        for line in lines:
            for span in line.get("spans", []):
                txt = str(span.get("text", "")).strip()
                if txt:
                    spans.append(txt)
        return " ".join(spans).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _clean_caption_text(text: str, max_chars: int = 320) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3] + "..."

    def _is_noise_text_block(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        page_height: float,
    ) -> bool:
        cleaned = self._normalize_text(text).lower()
        if not cleaned:
            return True

        # Typical running headers/footers in academic PDFs.
        top_band = bbox[1] <= page_height * 0.09
        bottom_band = bbox[3] >= page_height * 0.93
        if top_band or bottom_band:
            noise_terms = (
                "published as",
                "conference paper",
                "under review",
                "preprint",
                "arxiv",
                "copyright",
                "all rights reserved",
            )
            if any(term in cleaned for term in noise_terms):
                return True

        # Ignore standalone page numbers in footer/header areas.
        if top_band or bottom_band:
            compact = cleaned.replace(" ", "")
            if compact.isdigit() and len(compact) <= 4:
                return True

        return False

    @staticmethod
    def _sort_key(block: dict) -> tuple[float, float]:
        x0, y0, *_ = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
        return float(y0), float(x0)

    @staticmethod
    def _merge_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @staticmethod
    def _expand_bbox(
        bbox: tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        pad: float = 8.0,
    ) -> fitz.Rect:
        x0, y0, x1, y1 = bbox
        return fitz.Rect(
            max(0.0, x0 - pad),
            max(0.0, y0 - pad),
            min(page_width, x1 + pad),
            min(page_height, y1 + pad),
        )

    def _save_clip(self, page: fitz.Page, clip_rect: fitz.Rect, out_path: Path) -> None:
        matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False, clip=clip_rect)
        pixmap.save(out_path)

    def _merge_text_blocks(self, blocks: List[_TextBlock]) -> List[_TextBlock]:
        if not blocks:
            return []

        merged: List[_TextBlock] = []
        cur_text = ""
        cur_bbox: tuple[float, float, float, float] | None = None
        for block in blocks:
            if not cur_text:
                cur_text = block.text
                cur_bbox = block.bbox
                continue

            if len(cur_text) + 1 + len(block.text) <= self.text_chunk_chars:
                cur_text = f"{cur_text} {block.text}".strip()
                cur_bbox = self._merge_bbox(cur_bbox, block.bbox)  # type: ignore[arg-type]
            else:
                merged.append(_TextBlock(text=cur_text, bbox=cur_bbox))  # type: ignore[arg-type]
                cur_text = block.text
                cur_bbox = block.bbox

        if cur_text and cur_bbox is not None:
            merged.append(_TextBlock(text=cur_text, bbox=cur_bbox))
        return merged

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Keeps punctuation as standalone tokens to preserve local structure.
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _split_text_with_overlap(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        if not tokens:
            return []
        if len(tokens) <= self.text_chunk_tokens:
            return [" ".join(tokens)]

        chunks: List[str] = []
        step = max(1, self.text_chunk_tokens - self.text_overlap_tokens)
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + self.text_chunk_tokens)
            window = tokens[start:end]
            if not window:
                continue
            chunks.append(" ".join(window))
            if end >= len(tokens):
                break
        return chunks

    def _find_caption(self, image_bbox: tuple[float, float, float, float], text_blocks: List[_TextBlock]) -> str:
        _, y0, _, y1 = image_bbox
        keyword_candidates: list[tuple[float, int, str]] = []
        fallback_candidates: list[tuple[float, int, str]] = []
        keywords = ("figure", "fig.", "table", "图", "表")

        for block in text_blocks:
            block_text = self._normalize_text(block.text)
            if not block_text:
                continue
            by0, by1 = block.bbox[1], block.bbox[3]
            distance = 0.0
            is_near = False
            if by0 >= y1 and by0 - y1 <= self.caption_window:
                distance = by0 - y1
                is_near = True
            elif y0 >= by1 and y0 - by1 <= self.caption_window:
                distance = y0 - by1
                is_near = True

            if not is_near:
                continue

            lower = block_text.lower()
            has_keyword = any(key in lower for key in keywords)
            block_len = len(block_text)
            if has_keyword:
                keyword_candidates.append((distance, block_len, block_text))
            elif 16 <= block_len <= 360:
                fallback_candidates.append((distance, block_len, block_text))

        if keyword_candidates:
            keyword_candidates.sort(key=lambda item: (item[0], abs(item[1] - 120)))
            return self._clean_caption_text(keyword_candidates[0][2])
        if fallback_candidates:
            fallback_candidates.sort(key=lambda item: (item[0], abs(item[1] - 120)))
            return self._clean_caption_text(fallback_candidates[0][2])
        return ""

    def ingest(self, pdf_path: str | Path) -> tuple[str, int, List[ChunkRecord]]:
        source = Path(pdf_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        doc_id = make_doc_id(source)
        doc_cache = self.cache_root / doc_id
        pages_dir = doc_cache / "pages"
        chunks_dir = doc_cache / "chunks"
        pages_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks: List[ChunkRecord] = []

        with fitz.open(source) as pdf:
            page_count = pdf.page_count
            page_matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)

            for page_idx, page in enumerate(pdf, start=1):
                page_pixmap = page.get_pixmap(matrix=page_matrix, alpha=False)
                page_image_path = pages_dir / f"page_{page_idx:04d}.png"
                page_pixmap.save(page_image_path)

                page_rect = page.rect
                page_width, page_height = float(page_rect.width), float(page_rect.height)
                raw_blocks = page.get_text("dict").get("blocks", [])
                raw_blocks.sort(key=self._sort_key)

                text_blocks: List[_TextBlock] = []
                image_blocks: List[tuple[float, float, float, float]] = []

                for raw in raw_blocks:
                    bbox = tuple(float(v) for v in raw.get("bbox", [0, 0, 0, 0]))
                    if raw.get("type", 0) == 0:
                        text = self._block_text(raw)
                        if text and not self._is_noise_text_block(text, bbox, page_height):
                            text_blocks.append(_TextBlock(text=text, bbox=bbox))
                    elif raw.get("type", 0) == 1:
                        image_blocks.append(bbox)

                merged_text_chunks = self._merge_text_blocks(text_blocks)

                order = 0
                for t_idx, chunk in enumerate(merged_text_chunks, start=1):
                    clip_rect = self._expand_bbox(chunk.bbox, page_width, page_height)
                    image_path = chunks_dir / f"page_{page_idx:04d}_text_{t_idx:03d}.png"
                    self._save_clip(page, clip_rect, image_path)

                    split_texts = self._split_text_with_overlap(chunk.text)
                    if not split_texts:
                        split_texts = [chunk.text]

                    for text_part in split_texts:
                        chunk_id = f"p{page_idx:04d}_c{order:03d}"
                        chunks.append(
                            ChunkRecord(
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                page_num=page_idx,
                                order=order,
                                chunk_type="text",
                                text=text_part,
                                image_path=str(image_path),
                                bbox=[float(v) for v in chunk.bbox],
                            )
                        )
                        order += 1

                for i_idx, ibox in enumerate(image_blocks, start=1):
                    clip_rect = self._expand_bbox(ibox, page_width, page_height, pad=12.0)
                    image_path = chunks_dir / f"page_{page_idx:04d}_image_{i_idx:03d}.png"
                    self._save_clip(page, clip_rect, image_path)
                    caption = self._find_caption(ibox, text_blocks)

                    chunk_id = f"p{page_idx:04d}_c{order:03d}"
                    chunks.append(
                        ChunkRecord(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            page_num=page_idx,
                            order=order,
                            chunk_type="figure",
                            text=caption or "Figure/Table visual chunk",
                            image_path=str(image_path),
                            bbox=[float(v) for v in ibox],
                        )
                    )
                    order += 1

                # Ensure each page contributes at least one searchable chunk.
                if order == 0:
                    chunk_id = f"p{page_idx:04d}_c000"
                    chunks.append(
                        ChunkRecord(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            page_num=page_idx,
                            order=0,
                            chunk_type="page",
                            text=page.get_text("text") or "",
                            image_path=str(page_image_path),
                            bbox=[0.0, 0.0, page_width, page_height],
                        )
                    )

        return doc_id, page_count, chunks
