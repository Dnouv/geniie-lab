#!/usr/bin/env python
"""
Run a session experiment and plot its metrics in one command.

Example:
  python scripts/run_and_plot.py \\
      --topicset beir/dbpedia-entity/test \\
      --run-id 24 \\
      --tag mplq --tag inc --tag fqkr
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment and plot metrics.")
    parser.add_argument(
        "--run-script",
        default="scripts/run_session_experiment.py",
        help="Path to the run script (default: %(default)s).",
    )
    parser.add_argument(
        "--plot-script",
        default="scripts/plot_log_metrics.py",
        help="Path to the plot script (default: %(default)s).",
    )
    parser.add_argument(
        "--topicset",
        required=True,
        help="Topicset passed to the plot script (e.g., beir/dbpedia-entity/test).",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for stdout/stderr logs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="analyses",
        help="Base directory for plot outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        help="Full name for this run; overrides --run-id/--tag.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier (e.g., 24).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Short tag(s) to describe the run (e.g., mplq, inc, fqkr).",
    )
    parser.add_argument(
        "--plot-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args to pass to the plot script (put last).",
    )
    return parser.parse_args()


def build_name(run_id: str | None, tags: List[str]) -> str:
    if not run_id:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    tag_part = "_".join([t for t in tags if t])
    return f"{run_id}_{tag_part}" if tag_part else run_id


def main() -> None:
    args = parse_args()
    name = args.name or build_name(args.run_id, args.tag)

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir) / name
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / f"{name}.out"
    stderr_path = logs_dir / f"{name}.log"

    run_cmd = [sys.executable, args.run_script]
    plot_cmd = [
        sys.executable,
        args.plot_script,
        "--log",
        str(stdout_path),
        "--topicset",
        args.topicset,
        "--output-dir",
        str(output_dir),
    ]
    plot_cmd.extend(args.plot_args)

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        try:
            subprocess.run(run_cmd, stdout=stdout_file, stderr=stderr_file, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Run script failed (exit {exc.returncode}).", file=sys.stderr)
            print(f"[ERROR] stdout: {stdout_path}", file=sys.stderr)
            print(f"[ERROR] stderr: {stderr_path}", file=sys.stderr)
            try:
                tail = stderr_path.read_text(encoding="utf-8").splitlines()[-200:]
                if tail:
                    print("[ERROR] Last 200 lines from stderr:", file=sys.stderr)
                    print("\n".join(tail), file=sys.stderr)
            except Exception as read_exc:
                print(f"[ERROR] Failed to read stderr log: {read_exc}", file=sys.stderr)
            sys.exit(exc.returncode)

    subprocess.run(plot_cmd, check=True)
    print(f"Run complete.\n- stdout: {stdout_path}\n- stderr: {stderr_path}\n- plots: {output_dir}")


if __name__ == "__main__":
    main()
