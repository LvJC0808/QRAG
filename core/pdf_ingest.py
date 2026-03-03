from __future__ import annotations

from pathlib import Path
from typing import List

import fitz

from ..schemas import PageRecord
from .utils import make_doc_id


class PDFIngestor:
    def __init__(self, cache_root: Path, dpi: int = 180) -> None:
        self.cache_root = cache_root
        self.dpi = dpi

    def ingest(self, pdf_path: str | Path) -> tuple[str, List[PageRecord]]:
        source = Path(pdf_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        doc_id = make_doc_id(source)
        doc_cache = self.cache_root / doc_id / "pages"
        doc_cache.mkdir(parents=True, exist_ok=True)

        pages: List[PageRecord] = []
        with fitz.open(source) as pdf:
            matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            for idx, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = doc_cache / f"page_{idx:04d}.png"
                pixmap.save(image_path)

                pages.append(
                    PageRecord(
                        doc_id=doc_id,
                        page_num=idx,
                        text=text,
                        image_path=str(image_path),
                    )
                )

        return doc_id, pages
