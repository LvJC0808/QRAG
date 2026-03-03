import unittest

import numpy as np

from QRAG.core.dense_exact_index import DenseExactIndex


class DenseExactIndexTest(unittest.TestCase):
    def test_returns_expected_top_order(self):
        index = DenseExactIndex()
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        metadata = [
            {"page_num": 1, "text": "A", "image_path": "a.png"},
            {"page_num": 2, "text": "B", "image_path": "b.png"},
            {"page_num": 3, "text": "C", "image_path": "c.png"},
        ]
        index.build(embeddings, metadata)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        out = index.search(query, top_k=2)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].page_num, 1)
        self.assertEqual(out[1].page_num, 3)
        self.assertGreater(out[0].score, out[1].score)


if __name__ == "__main__":
    unittest.main()
