import unittest

import numpy as np

from QRAG.core.pipeline import QRAGPipeline
from QRAG.schemas import DimensionScores, JudgeResult, RetrievalCandidate


class FakeIngestor:
    def ingest(self, _):
        raise RuntimeError("not used in this test")


class FakeIndexStore:
    def save(self, **kwargs):
        return "/tmp/index"


class FakeEmbedder:
    model_name = "fake-embed"

    def embed_pages(self, pages, instruction):
        return np.zeros((len(pages), 4), dtype=np.float32)

    def embed_query(self, query, instruction):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeReranker:
    def rerank(self, query, candidates, instruction, top_n):
        out = []
        for idx, cand in enumerate(candidates[:top_n]):
            out.append((cand, 1.0 - idx * 0.1))
        return out


class FakeGenerator:
    def generate_answer(self, query, evidence, config, image_max_pixels):
        return "## 结论\n答案见 [p1]"

    def revise_answer(self, query, draft_answer, judge_feedback, evidence, config, image_max_pixels):
        return "## 结论\n修订答案见 [p1]"


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def evaluate(self, query, answer, evidence_items):
        self.calls += 1
        if self.calls == 1:
            return JudgeResult(
                overall_score=70,
                dimension_scores=DimensionScores(citation_validity=60),
                major_issues=["missing details"],
                actionable_feedback=["add citations"],
                verdict="revise",
                raw_text="",
            )
        return JudgeResult(
            overall_score=85,
            dimension_scores=DimensionScores(citation_validity=85),
            major_issues=[],
            actionable_feedback=[],
            verdict="accept",
            raw_text="",
        )


class PipelineSmokeTest(unittest.TestCase):
    def test_run_with_revision(self):
        pipeline = QRAGPipeline(
            ingestor=FakeIngestor(),
            index_store=FakeIndexStore(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            generator=FakeGenerator(),
            judge=FakeJudge(),
            retrieval_instruction="test",
        )

        pipeline.index.build(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            [{"page_num": 1, "text": "evidence", "image_path": "page1.png"}],
        )
        pipeline.state.ready = True

        bundle = pipeline.run("question")
        self.assertIn("修订", bundle.final_answer)
        self.assertEqual(bundle.judge_result.verdict, "accept")
        self.assertEqual(bundle.citations, [1])


if __name__ == "__main__":
    unittest.main()
