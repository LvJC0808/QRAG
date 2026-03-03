# QRAG

Local multimodal PDF RAG system based on:
- `Qwen3-VL-Embedding-8B` for recall
- `Qwen3-VL-Reranker-8B` for reranking
- `Qwen3-VL-8B-Instruct` for answer generation and LLM-as-a-Judge

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
- Index files are stored in `QRAG/data/indexes/<doc_id>/`.

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s QRAG/tests -p 'test_*.py'
```
