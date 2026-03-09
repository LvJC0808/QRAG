# QRAG

一个本地多模态 PDF RAG 系统，基于以下模型：
- `Qwen3-VL-Embedding-8B`：召回（recall）
- `Qwen3-VL-Reranker-8B`：重排序（rerank）
- `Qwen3-VL-8B-Instruct`：答案生成与 LLM-as-a-Judge 评估

当前检索器索引的是**页内切块**（文本块 + 视觉块），而不是整页。
每个切块都会保留页码、顺序、bbox 元数据，以及用于溯源回答的证据裁剪图。

## 运行

```bash
source .venv/bin/activate
python -m QRAG.app
```

## 说明

- 默认运行环境需要 4 张 GPU，服务绑定如下：
  - embed：`cuda:0`
  - rerank：`cuda:1`
  - generation：`cuda:2`
  - judge：`cuda:3`
- 可通过环境变量覆盖：
  - `QRAG_EMBED_DEVICE`、`QRAG_RERANK_DEVICE`、`QRAG_GEN_DEVICE`、`QRAG_JUDGE_DEVICE`
  - `QRAG_EMBED_BATCH_SIZE`（默认 `4`，若 ingest OOM 可降到 `1-2`）
  - `QRAG_INGEST_DPI`（默认 `160`，降低可减少视觉 token 压力）
  - `QRAG_INGEST_TEXT_CHARS`（默认 `900`，降低可缩短单块文本长度）
  - `QRAG_INGEST_TEXT_TOKENS`（默认 `220`）
  - `QRAG_INGEST_TEXT_OVERLAP`（默认 `40`）
- 索引文件存放在 `QRAG/data/indexes/<doc_id>/`。
- Tuning 页签支持快速模式，可减少网格组合数量并缩短运行时间。

## 测试

```bash
source .venv/bin/activate
python -m unittest discover -s QRAG/tests -p 'test_*.py'
```

## Benchmark Runner

使用内置 runner 可对固定查询集进行评测，并导出可对比指标。

```bash
source .venv/bin/activate
python -m QRAG.benchmarks.benchmark_runner \
  --pdf /root/shared-nvme/2601.04720v2.pdf \
  --samples /root/shared-nvme/QRAG/data/benchmark_samples/example.jsonl \
  --max-samples 10
```

输出结果写入 `QRAG/data/benchmarks/<run_id>/`：
- `summary.json`
- `per_sample.jsonl`

示例 jsonl 结构（每行一条记录）：

```json
{"id":"q1","query":"What is the core contribution?","gold_pages":[1,2],"gold_chunk_ids":["p1-c3"],"reference_answer":"..."}
```

### ViDoRe v3（计算机科学）数据转换

如果你已下载 `vidore_v3_computer_science`，可将 parquet 文件转换为按文档拆分的 jsonl：

```bash
source .venv/bin/activate
python -m QRAG.benchmarks.prepare_vidore_cs \
  --input-dir /root/shared-nvme/QRAG/vidore_v3_computer_science \
  --output-dir /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs \
  --min-score 1
```

然后按 PDF/文档运行 benchmark：

```bash
python -m QRAG.benchmarks.benchmark_runner \
  --pdf /root/shared-nvme/QRAG/vidore_v3_computer_science/pdfs/Introduction_to_Computer_Science.pdf \
  --samples /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs/Introduction_to_Computer_Science.jsonl \
  --max-samples 50
```

### 跨文档批量 benchmark

一条命令运行所有按文档划分的样本文件：

```bash
source .venv/bin/activate
python -m QRAG.benchmarks.run_batch_eval \
  --samples-dir /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs \
  --pdf-dir /root/shared-nvme/QRAG/vidore_v3_computer_science/pdfs \
  --max-samples-per-doc 30 \
  --disable-revision
```

快速一致性检查（仅校验 sample/pdf 匹配，不执行真实评测）：

```bash
python -m QRAG.benchmarks.run_batch_eval \
  --samples-dir /root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs \
  --pdf-dir /root/shared-nvme/QRAG/vidore_v3_computer_science/pdfs \
  --dry-run
```
