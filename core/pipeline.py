from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from ..config import GenerationDefaults, JudgeDefaults, RetrievalDefaults
from ..schemas import (
    AnswerBundle,
    EvidenceItem,
    GenerationConfig,
    PageRecord,
    PipelineState,
    RetrievalCandidate,
    RetrievalConfig,
)
from .dense_exact_index import DenseExactIndex
from .embedder import Qwen3VLEmbeddingService
from .generator import Qwen3VLGeneratorService
from .index_store import IndexStore
from .judge import Qwen3VLJudgeService
from .pdf_ingest import PDFIngestor
from .reranker import Qwen3VLRerankerService
from .utils import extract_citations, short_snippet


class QRAGPipeline:
    def __init__(
        self,
        ingestor: PDFIngestor,
        index_store: IndexStore,
        embedder: Qwen3VLEmbeddingService,
        reranker: Qwen3VLRerankerService,
        generator: Qwen3VLGeneratorService,
        judge: Qwen3VLJudgeService,
        retrieval_instruction: str,
        retrieval_defaults: Optional[RetrievalDefaults] = None,
        generation_defaults: Optional[GenerationDefaults] = None,
        judge_defaults: Optional[JudgeDefaults] = None,
    ) -> None:
        self.ingestor = ingestor
        self.index_store = index_store
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
        self.judge = judge
        self.retrieval_instruction = retrieval_instruction

        self.retrieval_defaults = retrieval_defaults or RetrievalDefaults()
        self.generation_defaults = generation_defaults or GenerationDefaults()
        self.judge_defaults = judge_defaults or JudgeDefaults()

        self.index = DenseExactIndex()
        self.state = PipelineState()
        self.last_bundle: AnswerBundle | None = None

    def default_retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            recall_top_k=self.retrieval_defaults.recall_top_k,
            rerank_top_n=self.retrieval_defaults.rerank_top_n,
            evidence_top_m=self.retrieval_defaults.evidence_top_m,
            image_max_pixels=self.retrieval_defaults.image_max_pixels,
        )

    def default_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.generation_defaults.max_new_tokens,
            temperature=self.generation_defaults.temperature,
            force_table=self.generation_defaults.force_table,
        )

    def ingest_pdf(self, pdf_path: str | Path) -> PipelineState:
        doc_id, pages = self.ingestor.ingest(pdf_path)
        embeddings = self.embedder.embed_pages(pages, instruction=self.retrieval_instruction)
        index_path = self.index_store.save(
            doc_id=doc_id,
            embeddings=embeddings,
            pages=pages,
            model_name=self.embedder.model_name,
        )

        self.index.build(embeddings, [page.model_dump() for page in pages])
        self.state = PipelineState(
            doc_id=doc_id,
            index_path=str(index_path),
            page_count=len(pages),
            ready=True,
        )
        return self.state

    def load_index(self, index_path: str | Path) -> PipelineState:
        embeddings, pages, manifest = self.index_store.load(index_path)
        self.index.build(embeddings, [item.model_dump() for item in pages])
        self.state = PipelineState(
            doc_id=str(manifest.get("doc_id", "unknown")),
            index_path=str(Path(index_path).resolve()),
            page_count=len(pages),
            ready=True,
        )
        return self.state

    def _build_evidence(
        self,
        reranked: List[tuple[RetrievalCandidate, float]],
        evidence_top_m: int,
    ) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        for candidate, rerank_score in reranked[: max(1, min(evidence_top_m, len(reranked)))]:
            evidence.append(
                EvidenceItem(
                    page_num=candidate.page_num,
                    image_path=candidate.image_path,
                    snippet=short_snippet(candidate.text, 500),
                    retrieval_score=float(candidate.score),
                    rerank_score=float(rerank_score),
                )
            )
        return evidence

    def should_revise(self, judge_score: float, citation_score: float) -> bool:
        return (
            judge_score < self.judge_defaults.score_threshold
            or citation_score < self.judge_defaults.citation_threshold
        )

    def run(
        self,
        query: str,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
        enable_revision: bool = True,
    ) -> AnswerBundle:
        if not self.state.ready or not self.index.is_ready():
            raise RuntimeError("Pipeline is not ready. Please ingest a PDF first.")

        r_cfg = retrieval_config or self.default_retrieval_config()
        g_cfg = generation_config or self.default_generation_config()

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(query, instruction=self.retrieval_instruction)
        timings["embedding_query_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        recall = self.index.search(query_vec, top_k=r_cfg.recall_top_k)
        timings["dense_retrieval_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        reranked = self.reranker.rerank(
            query=query,
            candidates=recall,
            instruction=self.retrieval_instruction,
            top_n=r_cfg.rerank_top_n,
        )
        timings["rerank_s"] = time.perf_counter() - t2

        evidence = self._build_evidence(reranked, evidence_top_m=r_cfg.evidence_top_m)

        t3 = time.perf_counter()
        draft = self.generator.generate_answer(
            query=query,
            evidence=evidence,
            config=g_cfg,
            image_max_pixels=r_cfg.image_max_pixels,
        )
        timings["generation_s"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        judge_result = self.judge.evaluate(query=query, answer=draft, evidence_items=evidence)
        timings["judge_s"] = time.perf_counter() - t4

        final_answer = draft

        if enable_revision and self.should_revise(
            judge_result.overall_score,
            judge_result.dimension_scores.citation_validity,
        ):
            t5 = time.perf_counter()
            revised = self.generator.revise_answer(
                query=query,
                draft_answer=draft,
                judge_feedback=judge_result.actionable_feedback,
                evidence=evidence,
                config=g_cfg,
                image_max_pixels=r_cfg.image_max_pixels,
            )
            timings["revision_s"] = time.perf_counter() - t5

            t6 = time.perf_counter()
            revised_judge = self.judge.evaluate(query=query, answer=revised, evidence_items=evidence)
            timings["judge_revision_s"] = time.perf_counter() - t6

            if revised_judge.overall_score >= judge_result.overall_score:
                final_answer = revised
                judge_result = revised_judge

        bundle = AnswerBundle(
            draft_answer=draft,
            final_answer=final_answer,
            citations=extract_citations(final_answer),
            evidence_items=evidence,
            judge_result=judge_result,
            timings=timings,
        )
        self.last_bundle = bundle
        return bundle
