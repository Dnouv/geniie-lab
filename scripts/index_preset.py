#!/usr/bin/env python
"""Index built-in dataset presets for BM25 or SPLADE."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


INDEX_PRESETS = {
    "touche2022": {
        "dataset": "argsme/2020-04-01/processed/touche-2022-task-1",
        "indices": {
            "bm25": "touche_2022_bm25",
            "splade": "touche_2022_splade",
        },
    },
    "trec-covid": {
        "dataset": "beir/trec-covid",
        "indices": {
            "bm25": "trec_covid_bm25",
            "splade": "trec_covid_splade",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a built-in dataset preset for BM25 or SPLADE."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(INDEX_PRESETS),
        required=True,
        help="Dataset preset to index.",
    )
    parser.add_argument(
        "--retrieval",
        choices=("bm25", "splade"),
        required=True,
        help="Retrieval flavor/index type.",
    )
    parser.add_argument("--host", default="localhost", help="OpenSearch host (default: %(default)s).")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port (default: %(default)s).")
    parser.add_argument("--no-ssl", action="store_true", help="Use HTTP instead of HTTPS.")
    parser.add_argument("--recreate", action="store_true", help="Delete the index if it exists.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Bulk batch size (defaults: 1000 for BM25, 200 for SPLADE).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="SPLADE encode batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="naver/splade-cocondenser-ensembledistil",
        help="SPLADE model name.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-k SPLADE terms (default: %(default)s).")
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=512,
        help="Max doc tokens (default: %(default)s).",
    )
    parser.add_argument("--device", default=None, help="Device override (e.g., cuda, mps, cpu).")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    preset = INDEX_PRESETS[args.preset]
    dataset = preset["dataset"]
    index_name = preset["indices"][args.retrieval]

    script_dir = Path(__file__).resolve().parent
    if args.retrieval == "bm25":
        script_name = "index_dataset.py"
        default_batch_size = 1000
    else:
        script_name = "index_splade.py"
        default_batch_size = 200

    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    script_path = script_dir / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        dataset,
        "--index",
        index_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--batch-size",
        str(batch_size),
    ]

    if args.no_ssl:
        cmd.append("--no-ssl")
    if args.recreate:
        cmd.append("--recreate")

    if args.retrieval == "splade":
        cmd.extend(
            [
                "--encode-batch-size",
                str(args.encode_batch_size),
                "--model",
                args.model,
                "--top-k",
                str(args.top_k),
                "--max-doc-length",
                str(args.max_doc_length),
            ]
        )
        if args.device:
            cmd.extend(["--device", args.device])

    return cmd


def main() -> None:
    args = parse_args()
    subprocess.run(build_command(args), check=True)


if __name__ == "__main__":
    main()
