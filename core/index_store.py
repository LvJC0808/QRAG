from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..schemas import ChunkRecord


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
        chunks: List[ChunkRecord],
        model_name: str,
    ) -> Path:
        target = self.index_dir(doc_id)
        target.mkdir(parents=True, exist_ok=True)

        emb_path = target / "embeddings.npy"
        meta_path = target / "metadata.jsonl"
        manifest_path = target / "manifest.json"

        np.save(emb_path, embeddings.astype(np.float16))

        with meta_path.open("w", encoding="utf-8") as fp:
            for item in chunks:
                fp.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

        manifest = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "page_count": len({item.page_num for item in chunks}),
            "embedding_dim": int(embeddings.shape[1]),
            "model_name": model_name,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load(self, index_path: str | Path) -> tuple[np.ndarray, List[ChunkRecord], dict]:
        source = Path(index_path).expanduser().resolve()
        emb_path = source / "embeddings.npy"
        meta_path = source / "metadata.jsonl"
        manifest_path = source / "manifest.json"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files missing under {source}")

        embeddings = np.load(emb_path)
        chunks: List[ChunkRecord] = []
        with meta_path.open("r", encoding="utf-8") as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "chunk_id" not in payload:
                    page_num = int(payload.get("page_num", 0))
                    payload["chunk_id"] = f"p{page_num:04d}_c{line_idx:03d}"
                if "order" not in payload:
                    payload["order"] = line_idx
                if "chunk_type" not in payload:
                    payload["chunk_type"] = "page"
                if "bbox" not in payload:
                    payload["bbox"] = []
                chunks.append(ChunkRecord.model_validate(payload))

        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Index metadata/embedding row mismatch under {source}: "
                f"{embeddings.shape[0]} vectors vs {len(chunks)} metadata rows. "
                "Please rebuild the index."
            )

        return embeddings, chunks, manifest
