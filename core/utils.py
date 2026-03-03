from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, List


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def make_doc_id(pdf_path: Path) -> str:
    digest = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:10]
    ts = int(time.time())
    return f"doc_{ts}_{digest}"


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def short_snippet(text: str, max_chars: int = 320) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def make_chunk_ref(page_num: int, order: int) -> str:
    return f"p{int(page_num)}-c{int(order)}"


def extract_citations(markdown_text: str) -> List[int]:
    text = markdown_text or ""
    hits = re.findall(r"\[p(\d+)(?:-c\d+)?\]", text)
    if not hits:
        hits = re.findall(r"\[p(\d+)\]", text)
    return sorted({int(item) for item in hits})


def extract_chunk_citations(markdown_text: str) -> List[str]:
    text = markdown_text or ""
    hits = re.findall(r"\[(p\d+-c\d+)\]", text)
    normalized = [item.lower() for item in hits]
    return sorted(set(normalized))


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
