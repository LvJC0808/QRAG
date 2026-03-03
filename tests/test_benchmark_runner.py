import unittest

from QRAG.benchmarks.benchmark_runner import canonical_chunk_id, parse_int_list, parse_str_list


class BenchmarkRunnerUtilsTest(unittest.TestCase):
    def test_canonical_chunk_id(self):
        self.assertEqual(canonical_chunk_id("p0003_c010"), "p3-c10")
        self.assertEqual(canonical_chunk_id("p3-c10"), "p3-c10")
        self.assertEqual(canonical_chunk_id("p03c001"), "p3-c1")

    def test_parse_helpers(self):
        self.assertEqual(parse_int_list("1, 2; 3"), [1, 2, 3])
        self.assertEqual(parse_int_list(["4", 5, "x"]), [4, 5])
        self.assertEqual(parse_str_list("a,b ; c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
