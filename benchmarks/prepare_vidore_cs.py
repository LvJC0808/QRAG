from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_parquet_concat(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def to_python_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def prepare(input_dir: Path, output_dir: Path, min_score: int = 1) -> dict:
    queries_path = input_dir / "queries" / "test-00000-of-00001.parquet"
    qrels_path = input_dir / "qrels" / "test-00000-of-00001.parquet"
    corpus_paths = sorted((input_dir / "corpus").glob("test-*.parquet"))

    if not queries_path.exists() or not qrels_path.exists() or not corpus_paths:
        raise FileNotFoundError("Missing required ViDoRe files (queries/qrels/corpus)")

    queries = pd.read_parquet(
        queries_path,
        columns=["query_id", "query", "answer", "language", "query_types", "content_type"],
    )
    qrels = pd.read_parquet(qrels_path, columns=["query_id", "corpus_id", "score"])
    qrels = qrels[qrels["score"] >= min_score]

    corpus = load_parquet_concat(
        corpus_paths,
        columns=["corpus_id", "doc_id", "page_number_in_doc"],
    )

    merged = qrels.merge(corpus, on="corpus_id", how="left")
    merged = merged.dropna(subset=["doc_id", "page_number_in_doc"]).copy()
    merged["page_number_in_doc"] = merged["page_number_in_doc"].astype(int)

    grouped = (
        merged.groupby(["query_id", "doc_id"], as_index=False)
        .agg(
            gold_pages=("page_number_in_doc", lambda x: sorted(set(int(v) for v in x))),
            positive_pairs=("corpus_id", "count"),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_doc_counts: dict[str, int] = {}

    for doc_id, sub in grouped.groupby("doc_id"):
        out_path = output_dir / f"{doc_id}.jsonl"
        per_doc_counts[doc_id] = len(sub)
        with out_path.open("w", encoding="utf-8") as fp:
            for _, row in sub.iterrows():
                query_row = queries.loc[queries["query_id"] == row["query_id"]].iloc[0]
                rec = {
                    "id": f"vidore_cs_q{int(row['query_id'])}_{doc_id}",
                    "query_id": int(row["query_id"]),
                    "doc_id": str(doc_id),
                    "query": str(query_row["query"]),
                    "reference_answer": str(query_row.get("answer", "") or ""),
                    "language": str(query_row.get("language", "")),
                    "query_types": [str(x) for x in to_python_list(query_row.get("query_types"))],
                    "content_type": [str(x) for x in to_python_list(query_row.get("content_type"))],
                    "gold_pages": [int(v) for v in row["gold_pages"]],
                    "gold_chunk_ids": [],
                    "metadata": {
                        "source": "vidore_v3_computer_science",
                        "positive_pairs": int(row["positive_pairs"]),
                        "min_score": int(min_score),
                    },
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "vidore_v3_computer_science",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "min_score": int(min_score),
        "num_queries_total": int(queries.shape[0]),
        "num_qrels_total": int(qrels.shape[0]),
        "num_query_doc_pairs": int(grouped.shape[0]),
        "per_doc_counts": per_doc_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ViDoRe v3 computer science benchmark files for QRAG")
    parser.add_argument(
        "--input-dir",
        default="/root/shared-nvme/QRAG/vidore_v3_computer_science",
        help="Path to vidore_v3_computer_science directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/shared-nvme/QRAG/data/benchmark_samples/vidore_v3_cs",
        help="Where converted jsonl files will be written",
    )
    parser.add_argument("--min-score", type=int, default=1, help="Minimum qrels score to keep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        min_score=args.min_score,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
