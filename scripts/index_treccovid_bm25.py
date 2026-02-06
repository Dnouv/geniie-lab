#!/usr/bin/env python
"""Index the TREC-COVID corpus for BM25."""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index BEIR TREC-COVID for BM25.")
    parser.add_argument("--host", default="localhost", help="OpenSearch host (default: %(default)s).")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port (default: %(default)s).")
    parser.add_argument("--no-ssl", action="store_true", help="Use HTTP instead of HTTPS.")
    parser.add_argument("--recreate", action="store_true", help="Delete the index if it exists.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Bulk batch size (default: %(default)s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/index_dataset.py",
        "--dataset",
        "beir/trec-covid",
        "--index",
        "trec_covid_bm25",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.no_ssl:
        cmd.append("--no-ssl")
    if args.recreate:
        cmd.append("--recreate")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
