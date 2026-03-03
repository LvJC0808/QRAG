import unittest

from QRAG.core.utils import extract_chunk_citations, extract_citations, make_chunk_ref


class CitationUtilsTest(unittest.TestCase):
    def test_extract_page_and_chunk_citations(self):
        text = "结论见 [p12-c3] 与 [p2-c10]，另外参考 [p12-c3]。"
        self.assertEqual(extract_citations(text), [2, 12])
        self.assertEqual(extract_chunk_citations(text), ["p12-c3", "p2-c10"])

    def test_make_chunk_ref(self):
        self.assertEqual(make_chunk_ref(7, 4), "p7-c4")


if __name__ == "__main__":
    unittest.main()
