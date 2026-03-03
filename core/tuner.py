from __future__ import annotations

import itertools
import json
import time
import uuid
from pathlib import Path
from typing import Iterable, List

from ..schemas import (
    DimensionScores,
    RetrievalConfig,
    TuningRunResult,
    TuningSummary,
)
from .pipeline import QRAGPipeline
from .utils import safe_mean


class JudgeTuner:
    def __init__(self, pipeline: QRAGPipeline, output_dir: Path) -> None:
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def objective(scores: DimensionScores) -> float:
        return (
            0.10 * scores.relevance
            + 0.35 * scores.groundedness
            + 0.25 * scores.completeness
            + 0.10 * scores.numeric_consistency
            + 0.20 * scores.citation_validity
        )

    @staticmethod
    def _aggregate(score_rows: List[DimensionScores]) -> DimensionScores:
        return DimensionScores(
            relevance=safe_mean([row.relevance for row in score_rows]),
            groundedness=safe_mean([row.groundedness for row in score_rows]),
            completeness=safe_mean([row.completeness for row in score_rows]),
            numeric_consistency=safe_mean([row.numeric_consistency for row in score_rows]),
            citation_validity=safe_mean([row.citation_validity for row in score_rows]),
        )

    def run_grid(
        self,
        queries: Iterable[str],
        image_max_pixels_grid: List[int] | None = None,
    ) -> TuningSummary:
        query_list = [q.strip() for q in queries if q.strip()]
        if not query_list:
            raise ValueError("At least one evaluation query is required")

        recall_grid = [12, 24, 36]
        rerank_grid = [4, 8, 12]
        evidence_grid = [3, 4, 5]
        pixel_grid = image_max_pixels_grid or [768 * 768, 1024 * 1024, 1280 * 1280]

        runs: List[TuningRunResult] = []
        for recall_top_k, rerank_top_n, evidence_top_m, image_max_pixels in itertools.product(
            recall_grid, rerank_grid, evidence_grid, pixel_grid
        ):
            cfg = RetrievalConfig(
                recall_top_k=recall_top_k,
                rerank_top_n=rerank_top_n,
                evidence_top_m=evidence_top_m,
                image_max_pixels=image_max_pixels,
            )

            dim_rows: List[DimensionScores] = []
            overall_rows: List[float] = []
            for q in query_list:
                bundle = self.pipeline.run(q, retrieval_config=cfg, enable_revision=True)
                dim_rows.append(bundle.judge_result.dimension_scores)
                overall_rows.append(bundle.judge_result.overall_score)

            avg_dims = self._aggregate(dim_rows)
            result = TuningRunResult(
                run_id=f"run_{uuid.uuid4().hex[:8]}",
                retrieval_config=cfg,
                avg_dimension_scores=avg_dims,
                objective_score=self.objective(avg_dims),
                mean_overall_score=safe_mean(overall_rows),
            )
            runs.append(result)

        runs.sort(key=lambda x: x.objective_score, reverse=True)
        best = runs[0]
        summary = TuningSummary(
            best_run_id=best.run_id,
            best_config=best.retrieval_config,
            best_objective_score=best.objective_score,
            runs=runs,
        )

        ts = int(time.time())
        out_path = self.output_dir / f"tuning_summary_{ts}.json"
        out_path.write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "best_config.json").write_text(
            json.dumps(best.retrieval_config.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def load_queries(self, source_path: str | Path) -> List[str]:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Query file not found: {source}")

        if source.suffix.lower() == ".jsonl":
            queries = []
            with source.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    q = str(row.get("query", "")).strip()
                    if q:
                        queries.append(q)
            return queries

        return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
