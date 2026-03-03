import unittest

from QRAG.core.judge import Qwen3VLJudgeService


class JudgeParseTest(unittest.TestCase):
    def test_parse_valid_json(self):
        raw = '''
        {
          "overall_score": 88,
          "dimension_scores": {
            "relevance": 90,
            "groundedness": 86,
            "completeness": 87,
            "numeric_consistency": 85,
            "citation_validity": 92
          },
          "major_issues": ["none"],
          "actionable_feedback": ["keep concise"],
          "verdict": "accept"
        }
        '''
        parsed = Qwen3VLJudgeService.parse_result(raw)
        self.assertEqual(parsed.verdict, "accept")
        self.assertAlmostEqual(parsed.dimension_scores.citation_validity, 92)

    def test_parse_invalid_json(self):
        parsed = Qwen3VLJudgeService.parse_result("not json")
        self.assertEqual(parsed.verdict, "revise")
        self.assertEqual(parsed.overall_score, 0.0)


if __name__ == "__main__":
    unittest.main()
