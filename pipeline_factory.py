from __future__ import annotations

from pathlib import Path

from .config import (
    CACHE_DIR,
    EMBED_MODEL_PATH,
    GEN_MODEL_PATH,
    INDEX_DIR,
    PROMPTS_DIR,
    RERANK_MODEL_PATH,
    RuntimeConfig,
    ensure_directories,
)
from .core.embedder import Qwen3VLEmbeddingService
from .core.generator import Qwen3VLGeneratorService
from .core.index_store import IndexStore
from .core.judge import Qwen3VLJudgeService
from .core.pdf_ingest import PDFIngestor
from .core.pipeline import QRAGPipeline
from .core.reranker import Qwen3VLRerankerService
from .core.utils import read_text_file


def create_pipeline(runtime: RuntimeConfig | None = None) -> QRAGPipeline:
    ensure_directories()
    runtime = runtime or RuntimeConfig()

    generation_template = read_text_file(PROMPTS_DIR / "generation_v1.txt")
    revision_template = read_text_file(PROMPTS_DIR / "revision_v1.txt")
    judge_template = read_text_file(PROMPTS_DIR / "judge_v1.txt")
    retrieval_instruction = read_text_file(PROMPTS_DIR / "retrieval_instruction_v1.txt")

    ingestor = PDFIngestor(
        cache_root=CACHE_DIR,
        dpi=runtime.ingest_dpi,
        text_chunk_chars=runtime.ingest_text_chunk_chars,
        text_chunk_tokens=runtime.ingest_text_chunk_tokens,
        text_overlap_tokens=runtime.ingest_text_overlap_tokens,
    )
    index_store = IndexStore(root_dir=INDEX_DIR)

    embedder = Qwen3VLEmbeddingService(
        model_path=Path(EMBED_MODEL_PATH),
        device=runtime.embed_device,
        dtype=runtime.torch_dtype,
        attn_implementation=runtime.attn_implementation,
        batch_size=runtime.embed_batch_size,
    )
    reranker = Qwen3VLRerankerService(
        model_path=Path(RERANK_MODEL_PATH),
        device=runtime.rerank_device,
        dtype=runtime.torch_dtype,
        attn_implementation=runtime.attn_implementation,
    )
    generator = Qwen3VLGeneratorService(
        model_path=Path(GEN_MODEL_PATH),
        device=runtime.gen_device,
        dtype=runtime.torch_dtype,
        attn_implementation=runtime.attn_implementation,
        prompt_template=generation_template,
        revision_template=revision_template,
    )
    judge = Qwen3VLJudgeService(
        model_path=Path(GEN_MODEL_PATH),
        device=runtime.judge_device,
        dtype=runtime.torch_dtype,
        attn_implementation=runtime.attn_implementation,
        judge_template=judge_template,
    )

    return QRAGPipeline(
        ingestor=ingestor,
        index_store=index_store,
        embedder=embedder,
        reranker=reranker,
        generator=generator,
        judge=judge,
        retrieval_instruction=retrieval_instruction,
    )
