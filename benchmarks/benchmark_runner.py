from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import DATA_DIR
from ..core.utils import make_chunk_ref
from ..pipeline_factory import create_pipeline
from ..schemas import GenerationConfig, RetrievalConfig


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    query: str
    gold_pages: list[int]
    gold_chunk_ids: list[str]
    reference_answer: str
    metadata: dict[str, Any]


def canonical_chunk_id(value: str) -> str:
    val = (value or "").strip().lower()
    if not val:
        return ""

    import re

    m = re.match(r"p0*(\d+)_c0*(\d+)$", val)
    if m:
        return f"p{int(m.group(1))}-c{int(m.group(2))}"

    m = re.match(r"p0*(\d+)-c0*(\d+)$", val)
    if m:
        return f"p{int(m.group(1))}-c{int(m.group(2))}"

    m = re.match(r"p0*(\d+)c0*(\d+)$", val)
    if m:
        return f"p{int(m.group(1))}-c{int(m.group(2))}"

    return val


def parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        out = []
        for part in parts:
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                continue
        return out
    return []


def parse_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return [part for part in parts if part]
    return []


def load_jsonl_samples(
    source_path: str | Path,
    query_field: str = "query",
    id_field: str = "id",
    gold_pages_field: str = "gold_pages",
    gold_chunks_field: str = "gold_chunk_ids",
    reference_field: str = "reference_answer",
) -> list[BenchmarkSample]:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Sample file not found: {source}")

    samples: list[BenchmarkSample] = []
    with source.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query = str(payload.get(query_field, "")).strip()
            if not query:
                continue

            sample_id = str(payload.get(id_field, f"sample_{idx + 1:04d}"))
            gold_pages = parse_int_list(payload.get(gold_pages_field))
            gold_chunk_ids = [canonical_chunk_id(x) for x in parse_str_list(payload.get(gold_chunks_field))]
            reference_answer = str(payload.get(reference_field, "")).strip()

            metadata = {
                key: value
                for key, value in payload.items()
                if key
                not in {query_field, id_field, gold_pages_field, gold_chunks_field, reference_field}
            }
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    query=query,
                    gold_pages=gold_pages,
                    gold_chunk_ids=gold_chunk_ids,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
            )
    if not samples:
        raise ValueError("No valid benchmark samples found in input file")
    return samples


def evidence_chunk_refs(bundle) -> set[str]:
    refs = set()
    for item in bundle.evidence_items:
        if item.chunk_id:
            refs.add(canonical_chunk_id(item.chunk_id))
        refs.add(canonical_chunk_id(make_chunk_ref(item.page_num, item.order)))
    return refs


def evidence_pages(bundle) -> set[int]:
    return {int(item.page_num) for item in bundle.evidence_items}


def compute_sample_metrics(bundle, sample: BenchmarkSample) -> dict[str, Any]:
    dims = bundle.judge_result.dimension_scores

    sample_row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "query": sample.query,
        "overall_score": float(bundle.judge_result.overall_score),
        "relevance": float(dims.relevance),
        "groundedness": float(dims.groundedness),
        "completeness": float(dims.completeness),
        "numeric_consistency": float(dims.numeric_consistency),
        "citation_validity": float(dims.citation_validity),
        "verdict": bundle.judge_result.verdict,
        "citation_count": len(bundle.citations),
        "chunk_citation_count": len(bundle.citation_chunk_ids),
        "evidence_count": len(bundle.evidence_items),
        "timings": bundle.timings,
    }

    if sample.gold_pages:
        pages = evidence_pages(bundle)
        sample_row["hit_gold_page"] = bool(pages.intersection(set(sample.gold_pages)))
    else:
        sample_row["hit_gold_page"] = None

    if sample.gold_chunk_ids:
        refs = evidence_chunk_refs(bundle)
        sample_row["hit_gold_chunk"] = bool(refs.intersection(set(sample.gold_chunk_ids)))
    else:
        sample_row["hit_gold_chunk"] = None

    if sample.reference_answer:
        # Placeholder text-overlap metric for optional human reference. Keeping lightweight and deterministic.
        answer_tokens = set(bundle.final_answer.lower().split())
        ref_tokens = set(sample.reference_answer.lower().split())
        denom = max(1, len(ref_tokens))
        sample_row["reference_token_recall"] = len(answer_tokens.intersection(ref_tokens)) / denom
    else:
        sample_row["reference_token_recall"] = None

    sample_row["final_answer"] = bundle.final_answer
    sample_row["draft_answer"] = bundle.draft_answer
    sample_row["citation_chunk_ids"] = bundle.citation_chunk_ids
    return sample_row


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def avg(field: str) -> float:
        vals = [float(row[field]) for row in rows if row.get(field) is not None]
        return statistics.fmean(vals) if vals else 0.0

    def ratio_true(field: str) -> float | None:
        vals = [row[field] for row in rows if row.get(field) is not None]
        if not vals:
            return None
        return sum(1 for item in vals if item) / len(vals)

    latency_totals = [sum(float(v) for v in row.get("timings", {}).values()) for row in rows]

    summary = {
        "num_samples": len(rows),
        "avg_overall_score": avg("overall_score"),
        "avg_relevance": avg("relevance"),
        "avg_groundedness": avg("groundedness"),
        "avg_completeness": avg("completeness"),
        "avg_numeric_consistency": avg("numeric_consistency"),
        "avg_citation_validity": avg("citation_validity"),
        "avg_citation_count": avg("citation_count"),
        "avg_chunk_citation_count": avg("chunk_citation_count"),
        "gold_page_hit_rate": ratio_true("hit_gold_page"),
        "gold_chunk_hit_rate": ratio_true("hit_gold_chunk"),
        "avg_reference_token_recall": avg("reference_token_recall"),
        "avg_total_latency_s": statistics.fmean(latency_totals) if latency_totals else 0.0,
        "p95_total_latency_s": sorted(latency_totals)[max(0, int(len(latency_totals) * 0.95) - 1)]
        if latency_totals
        else 0.0,
        "args": vars(args),
    }
    return summary


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: Iterable[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "per_sample.jsonl"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with sample_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_jsonl_samples(
        args.samples,
        query_field=args.query_field,
        id_field=args.id_field,
        gold_pages_field=args.gold_pages_field,
        gold_chunks_field=args.gold_chunks_field,
        reference_field=args.reference_field,
    )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    pipeline = create_pipeline()

    if args.index:
        pipeline.load_index(args.index)
    elif args.pdf:
        pipeline.ingest_pdf(args.pdf)
    else:
        raise ValueError("Either --pdf or --index must be provided")

    r_cfg = RetrievalConfig(
        recall_top_k=args.recall_top_k,
        rerank_top_n=args.rerank_top_n,
        evidence_top_m=args.evidence_top_m,
        image_max_pixels=args.image_max_pixels,
    )
    g_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        force_table=args.force_table,
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, sample in enumerate(samples, start=1):
        bundle = pipeline.run(
            sample.query,
            retrieval_config=r_cfg,
            generation_config=g_cfg,
            enable_revision=not args.disable_revision,
        )
        row = compute_sample_metrics(bundle, sample)
        row["sample_index"] = idx
        rows.append(row)

    summary = summarize(rows, args)
    summary["wall_time_s"] = time.perf_counter() - started

    run_id = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_dir = Path(args.output_dir).expanduser().resolve() / run_id
    write_outputs(out_dir, summary, rows)

    summary["run_id"] = run_id
    summary["output_dir"] = str(out_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QRAG benchmark on a local sample set")
    parser.add_argument("--samples", required=True, help="Path to benchmark jsonl file")
    parser.add_argument("--pdf", default="", help="PDF path to ingest before benchmarking")
    parser.add_argument("--index", default="", help="Existing index path (skip ingest)")
    parser.add_argument("--output-dir", default=str(Path(DATA_DIR) / "benchmarks"), help="Benchmark output root")

    parser.add_argument("--query-field", default="query")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--gold-pages-field", default="gold_pages")
    parser.add_argument("--gold-chunks-field", default="gold_chunk_ids")
    parser.add_argument("--reference-field", default="reference_answer")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")

    parser.add_argument("--recall-top-k", type=int, default=24)
    parser.add_argument("--rerank-top-n", type=int, default=8)
    parser.add_argument("--evidence-top-m", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 1024)

    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--force-table", dest="force_table", action="store_true")
    parser.add_argument("--no-force-table", dest="force_table", action="store_false")
    parser.set_defaults(force_table=True)
    parser.add_argument("--disable-revision", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.pdf and not args.index:
        parser.error("You must pass one of --pdf or --index")

    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
