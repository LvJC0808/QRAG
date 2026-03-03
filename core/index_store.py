from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..schemas import PageRecord


class IndexStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def index_dir(self, doc_id: str) -> Path:
        return self.root_dir / doc_id

    def save(
        self,
        doc_id: str,
        embeddings: np.ndarray,
        pages: List[PageRecord],
        model_name: str,
    ) -> Path:
        target = self.index_dir(doc_id)
        target.mkdir(parents=True, exist_ok=True)

        emb_path = target / "embeddings.npy"
        meta_path = target / "metadata.jsonl"
        manifest_path = target / "manifest.json"

        np.save(emb_path, embeddings.astype(np.float16))

        with meta_path.open("w", encoding="utf-8") as fp:
            for item in pages:
                fp.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

        manifest = {
            "doc_id": doc_id,
            "page_count": len(pages),
            "embedding_dim": int(embeddings.shape[1]),
            "model_name": model_name,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load(self, index_path: str | Path) -> tuple[np.ndarray, List[PageRecord], dict]:
        source = Path(index_path).expanduser().resolve()
        emb_path = source / "embeddings.npy"
        meta_path = source / "metadata.jsonl"
        manifest_path = source / "manifest.json"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files missing under {source}")

        embeddings = np.load(emb_path)
        pages: List[PageRecord] = []
        with meta_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                pages.append(PageRecord.model_validate_json(line))

        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return embeddings, pages, manifest
