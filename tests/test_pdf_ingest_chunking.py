import tempfile
import unittest
from pathlib import Path

import fitz

from QRAG.core.pdf_ingest import PDFIngestor


class PDFIngestChunkingTest(unittest.TestCase):
    def test_ingest_produces_chunk_records(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            pdf_path = tmpdir / "sample.pdf"

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Chunk one. " * 80)
            page.insert_text((72, 140), "Chunk two. " * 80)
            doc.save(pdf_path)
            doc.close()

            ingestor = PDFIngestor(cache_root=tmpdir / "cache", dpi=96, text_chunk_chars=350)
            doc_id, page_count, chunks = ingestor.ingest(pdf_path)

            self.assertTrue(doc_id.startswith("doc_"))
            self.assertEqual(page_count, 1)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertEqual(chunks[0].page_num, 1)
            self.assertTrue(chunks[0].chunk_id.startswith("p0001_c"))
            self.assertIn(chunks[0].chunk_type, {"text", "figure", "page"})


if __name__ == "__main__":
    unittest.main()
