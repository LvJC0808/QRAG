# QRAG

Local multimodal PDF RAG system based on:
- `Qwen3-VL-Embedding-8B` for recall
- `Qwen3-VL-Reranker-8B` for reranking
- `Qwen3-VL-8B-Instruct` for answer generation and LLM-as-a-Judge

The current retriever indexes **intra-page chunks** (text blocks + visual blocks) instead of whole pages.
Each chunk keeps page id, order, bbox metadata, and a cropped evidence image for grounded answers.

## Run

```bash
source .venv/bin/activate
python -m QRAG.app
```

## Notes

- Default runtime expects 4 GPUs and binds services to:
  - embed: `cuda:0`
  - rerank: `cuda:1`
  - generation: `cuda:2`
  - judge: `cuda:3`
- You can override with env vars:
  - `QRAG_EMBED_DEVICE`, `QRAG_RERANK_DEVICE`, `QRAG_GEN_DEVICE`, `QRAG_JUDGE_DEVICE`
  - `QRAG_EMBED_BATCH_SIZE` (default `4`, lower it to `1-2` if ingest OOMs)
  - `QRAG_INGEST_DPI` (default `160`, lower it for less visual-token pressure)
  - `QRAG_INGEST_TEXT_CHARS` (default `900`, lowers per-chunk text length)
  - `QRAG_INGEST_TEXT_TOKENS` (default `220`)
  - `QRAG_INGEST_TEXT_OVERLAP` (default `40`)
- Index files are stored in `QRAG/data/indexes/<doc_id>/`.
- Tuning tab supports a quick mode to reduce grid combinations and runtime.

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s QRAG/tests -p 'test_*.py'
```

## Benchmark Runner

Use the built-in runner to evaluate fixed query sets and export comparable metrics.

```bash
source .venv/bin/activate
python -m QRAG.benchmarks.benchmark_runner \
  --pdf /root/shared-nvme/2601.04720v2.pdf \
  --samples /root/shared-nvme/QRAG/data/benchmark_samples/example.jsonl \
  --max-samples 10
```

Outputs are written to `QRAG/data/benchmarks/<run_id>/`:
- `summary.json`
- `per_sample.jsonl`

Sample jsonl schema (one record per line):

```json
{"id":"q1","query":"What is the core contribution?","gold_pages":[1,2],"gold_chunk_ids":["p1-c3"],"reference_answer":"..."}
```

### ViDoRe v3 (Computer Science) conversion

If you downloaded `vidore_v3_computer_science`, convert parquet files to per-document jsonl:

```bash
source .venv/bin/activate
python -m QRAG.benchmarks.prepare_vidore_cs \
  --input-dir /root/shared-nvme/QRAG/vidore_v3_computer_science \
  --output-dir /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs \
  --min-score 1
```

Then run benchmark per PDF/doc:

```bash
python -m QRAG.benchmarks.benchmark_runner \
  --pdf /root/shared-nvme/QRAG/vidore_v3_computer_science/pdfs/Introduction_to_Computer_Science.pdf \
  --samples /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs/Introduction_to_Computer_Science.jsonl \
  --max-samples 50
```
