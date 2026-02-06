#!/usr/bin/env python
"""Index the TREC-COVID corpus for SPLADE."""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index BEIR TREC-COVID for SPLADE.")
    parser.add_argument("--host", default="localhost", help="OpenSearch host (default: %(default)s).")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port (default: %(default)s).")
    parser.add_argument("--no-ssl", action="store_true", help="Use HTTP instead of HTTPS.")
    parser.add_argument("--recreate", action="store_true", help="Delete the index if it exists.")
    parser.add_argument("--batch-size", type=int, default=200, help="Bulk batch size (default: %(default)s).")
    parser.add_argument("--encode-batch-size", type=int, default=8, help="SPLADE encode batch size (default: %(default)s).")
    parser.add_argument("--model", default="naver/splade-cocondenser-ensembledistil", help="SPLADE model name.")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k SPLADE terms (default: %(default)s).")
    parser.add_argument("--max-doc-length", type=int, default=512, help="Max doc tokens (default: %(default)s).")
    parser.add_argument("--device", default=None, help="Device override (e.g., cuda, mps, cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/index_splade.py",
        "--dataset",
        "beir/trec-covid",
        "--index",
        "trec_covid_splade",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--batch-size",
        str(args.batch_size),
        "--encode-batch-size",
        str(args.encode_batch_size),
        "--model",
        args.model,
        "--top-k",
        str(args.top_k),
        "--max-doc-length",
        str(args.max_doc_length),
    ]
    if args.no_ssl:
        cmd.append("--no-ssl")
    if args.recreate:
        cmd.append("--recreate")
    if args.device:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
