from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    recall_top_k: int = Field(default=24, ge=1, le=200)
    rerank_top_n: int = Field(default=8, ge=1, le=100)
    evidence_top_m: int = Field(default=4, ge=1, le=20)
    image_max_pixels: int = Field(default=1024 * 1024, ge=256 * 256)


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=1200, ge=64, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    force_table: bool = True


class PageRecord(BaseModel):
    doc_id: str
    page_num: int
    text: str
    image_path: str


class ChunkRecord(BaseModel):
    doc_id: str
    chunk_id: str
    page_num: int
    order: int
    chunk_type: str = "text"
    text: str
    image_path: str
    bbox: List[float] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    chunk_id: str = ""
    page_num: int
    order: int = 0
    chunk_type: str = "text"
    text: str
    image_path: str
    bbox: List[float] = Field(default_factory=list)
    score: float


class EvidenceItem(BaseModel):
    chunk_id: str = ""
    page_num: int
    order: int = 0
    chunk_type: str = "text"
    image_path: str
    snippet: str
    bbox: List[float] = Field(default_factory=list)
    retrieval_score: float
    rerank_score: float


class DimensionScores(BaseModel):
    relevance: float = Field(default=0.0, ge=0.0, le=100.0)
    groundedness: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness: float = Field(default=0.0, ge=0.0, le=100.0)
    numeric_consistency: float = Field(default=0.0, ge=0.0, le=100.0)
    citation_validity: float = Field(default=0.0, ge=0.0, le=100.0)


class JudgeResult(BaseModel):
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    major_issues: List[str] = Field(default_factory=list)
    actionable_feedback: List[str] = Field(default_factory=list)
    verdict: str = Field(default="revise")
    raw_text: str = ""


class AnswerBundle(BaseModel):
    draft_answer: str
    final_answer: str
    citations: List[int]
    citation_chunk_ids: List[str] = Field(default_factory=list)
    evidence_items: List[EvidenceItem]
    judge_result: JudgeResult
    timings: Dict[str, float]


class PipelineState(BaseModel):
    doc_id: Optional[str] = None
    index_path: Optional[str] = None
    page_count: int = 0
    chunk_count: int = 0
    ready: bool = False


class TuningRunResult(BaseModel):
    run_id: str
    retrieval_config: RetrievalConfig
    avg_dimension_scores: DimensionScores
    objective_score: float
    mean_overall_score: float


class TuningSummary(BaseModel):
    best_run_id: str
    best_config: RetrievalConfig
    best_objective_score: float
    runs: List[TuningRunResult]
