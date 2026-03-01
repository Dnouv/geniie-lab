#!/usr/bin/env python
"""
Summarize ranking metrics from JSONL experiment logs into markdown tables.

Outputs:
1) Dataset × Ranker × Policy table (mean ± std across topics).
2) Ranker × Policy summary (dataset means, datasets weighted equally).
3) Dataset × Policy (ranker means, rankers weighted equally).
4) Policy overall (dataset-ranker means, all equally weighted).
5) Dataset × Ranker (policy means, policies weighted equally).

Example:
  python scripts/summarize_metrics_tables.py \\
    --log logs/final/dbpedia/oss120_bm25_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_bm25_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_bm25_fqkr_2.out --label forget_queries_keep_reason \\
    --log logs/final/dbpedia/oss120_splade_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_splade_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_splade_fqkr_0.out --label forget_queries_keep_reason \\
    --metrics "CumRecall,CumRecall@100,nDCG@10,RR@10,R@100" \\
    --precision-csv analyses/avg/all_datasets_avg_cumprecision_grid_metrics/per_topic_iteration_values.csv \\
    --output analyses/summary/metrics_tables.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics into markdown tables.")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path to a JSONL log file (repeat for multiple policies/rankers).",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for each --log (memory policy name).",
    )
    parser.add_argument(
        "--metrics",
        default="CumRecall,CumRecall@100,nDCG@10,RR@10,R@100",
        help="Comma-separated metric keys to summarize.",
    )
    parser.add_argument(
        "--precision-csv",
        action="append",
        default=None,
        help=(
            "Path to precision CSV exported by plot_cumprecision_avg_grid.py "
            "(per_topic_iteration_values.csv). Can be repeated."
        ),
    )
    parser.add_argument(
        "--precision-metrics",
        default="CumPrecision,CumPrecision@100",
        help=(
            "Comma-separated precision metric keys to merge from --precision-csv. "
            "These are auto-appended to --metrics when --precision-csv is provided."
        ),
    )
    parser.add_argument(
        "--policy-label-from",
        choices=["auto", "label", "suffix", "prefix"],
        default="auto",
        help="How to derive policy name from --label (default: %(default)s).",
    )
    parser.add_argument(
        "--policy-suffix-sep",
        default="_",
        help="Separator for suffix-based policy extraction (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output markdown file path.",
    )
    return parser.parse_args()


def mean_std(values: List[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def load_iteration_metrics(
    log_path: str,
    metrics: List[str],
) -> tuple[str, str, Dict[str, Dict[str, List[float]]]]:
    per_topic_iters: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ranker = "unknown"
    dataset = "unknown"

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("stage") != "ranking":
                continue
            topic = entry.get("topic_id", "<unknown>")
            ranker = entry.get("ranker") or ranker
            dataset = entry.get("dataset") or dataset
            perf = entry.get("performance") or {}
            for m in metrics:
                value = perf.get(m)
                if value is None:
                    continue
                try:
                    per_topic_iters[topic][m].append(float(value))
                except (TypeError, ValueError):
                    continue

    return ranker, dataset, per_topic_iters


def path_lookup_keys(path_text: str) -> Set[str]:
    p = Path(path_text).expanduser()
    keys = {os.path.normpath(str(p))}
    try:
        keys.add(os.path.normpath(str(p.resolve())))
    except Exception:
        pass
    return keys


def load_precision_csv_values(
    csv_paths: List[str],
    allowed_metrics: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    # log_path_key -> topic_id -> metric -> list[per-iteration value]
    staged: Dict[str, Dict[str, Dict[str, List[Tuple[int, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"log_path", "topic_id", "metric", "iteration", "value"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise SystemExit(
                    f"Precision CSV '{csv_path}' missing required columns: {sorted(required)}"
                )
            for row in reader:
                metric = (row.get("metric") or "").strip()
                if not metric:
                    continue
                if allowed_metrics is not None and metric not in allowed_metrics:
                    continue
                topic_id = (row.get("topic_id") or "").strip()
                log_path = (row.get("log_path") or "").strip()
                if not topic_id or not log_path:
                    continue
                try:
                    iteration = int(row.get("iteration", ""))
                    value = float(row.get("value", ""))
                except (TypeError, ValueError):
                    continue
                for key in path_lookup_keys(log_path):
                    staged[key][topic_id][metric].append((iteration, value))

    out: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(dict))
    for log_key, topic_map in staged.items():
        for topic_id, metric_map in topic_map.items():
            for metric, pairs in metric_map.items():
                # Deduplicate by iteration; later rows overwrite earlier rows.
                it_to_val: Dict[int, float] = {}
                for it, val in pairs:
                    it_to_val[it] = val
                out[log_key][topic_id][metric] = [it_to_val[it] for it in sorted(it_to_val)]
    return out


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.4f} ± {std:.4f}"


def compute_stats(
    values_map: Dict,
    metrics: List[str],
) -> Dict:
    stats: Dict = {}
    for key, metric_map in values_map.items():
        stats[key] = {}
        for m in metrics:
            stats[key][m] = mean_std(metric_map.get(m, []))
    return stats


def render_table(
    title: str,
    headers_prefix: List[str],
    metrics: List[str],
    stats: Dict,
    bold_max: bool = False,
) -> List[str]:
    lines: List[str] = []
    lines.append(title)
    lines.append("")
    header = headers_prefix + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    max_by_metric: Dict[str, float] = {}
    if bold_max:
        for m in metrics:
            candidates = [stats[key][m][0] for key in stats if stats[key][m][0] is not None]
            max_by_metric[m] = max(candidates, default=None)

    for raw_key in sorted(stats.keys()):
        key_tuple = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        row = list(key_tuple)
        for m in metrics:
            mean, std = stats[raw_key][m]
            cell = format_mean_std(mean, std)
            if (
                bold_max
                and mean is not None
                and max_by_metric.get(m, None) is not None
                and math.isclose(mean, max_by_metric[m], rel_tol=1e-9, abs_tol=1e-12)
            ):
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if len(args.log) != len(args.label):
        raise SystemExit("Number of --log and --label arguments must match.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")
    precision_metrics = [m.strip() for m in args.precision_metrics.split(",") if m.strip()]
    if args.precision_csv:
        for metric in precision_metrics:
            if metric not in metrics:
                metrics.append(metric)

    precision_by_log_topic_metric: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    if args.precision_csv:
        precision_by_log_topic_metric = load_precision_csv_values(
            args.precision_csv,
            allowed_metrics=set(precision_metrics),
        )

    # (dataset, ranker, policy) -> metric -> list of per-topic iteration means
    drp_topic_means: Dict[tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def derive_policy(label: str) -> str:
        known = [
            "forget_queries_keep_reason",
            "forget_queries_half",
            "forget_queries",
            "full",
        ]
        if args.policy_label_from == "auto":
            for key in known:
                if key in label:
                    return key
        if args.policy_label_from == "label":
            return label
        parts = label.split(args.policy_suffix_sep)
        if args.policy_label_from == "suffix":
            return parts[-1] if parts else label
        return parts[0] if parts else label

    for log_path, label in zip(args.log, args.label):
        ranker, dataset, per_topic_iters = load_iteration_metrics(log_path, metrics)
        precision_payload = None
        for key in path_lookup_keys(log_path):
            if key in precision_by_log_topic_metric:
                precision_payload = precision_by_log_topic_metric[key]
                break
        if precision_payload:
            # Precision CSV is reconstructed from rel_judge and should override log values.
            for topic_id, metric_map in precision_payload.items():
                for metric, values in metric_map.items():
                    if metric not in metrics or not values:
                        continue
                    per_topic_iters[topic_id][metric] = list(values)

        policy = derive_policy(label)
        condition_key = (dataset, ranker, policy)
        for topic_id, metric_iters in per_topic_iters.items():
            for m, values in metric_iters.items():
                if not values:
                    continue
                mean_val, _ = mean_std(values)
                if mean_val is None:
                    continue
                drp_topic_means[condition_key][m].append(mean_val)

    # Build markdown
    out_lines: List[str] = []
    out_lines.append("# Metric Summary Tables")
    out_lines.append("")
    out_lines.append("## Definitions and Formulas")
    out_lines.append("")
    out_lines.append("- Topic-level iteration aggregate:")
    out_lines.append("  - For each topic and metric, we first average over iterations present in the log.")
    out_lines.append("  - Formula: `topic_mean = (1/T) * sum_{i=1..T} metric_i`")
    out_lines.append("- Condition key:")
    out_lines.append("  - A condition is `(dataset, ranker, policy)`.")
    out_lines.append("- Table cell format:")
    out_lines.append("  - Each cell is reported as `mean +- sample_std` over the values listed in that table definition.")
    out_lines.append("")
    out_lines.append("### Table Definitions")
    out_lines.append("")
    out_lines.append("- `Summary (Ranker x Policy, Iterations Aggregated)`")
    out_lines.append("  - Inputs: condition means grouped by `(ranker, policy)` across datasets.")
    out_lines.append("  - Weighting: each dataset-condition contributes equally.")
    out_lines.append("- `Per-Condition (Dataset x Ranker x Policy), Iterations Aggregated`")
    out_lines.append("  - Inputs: topic means inside each `(dataset, ranker, policy)`.")
    out_lines.append("  - Weighting: each topic contributes equally within that condition.")
    out_lines.append("- `Per-Condition (Dataset x Policy, Rankers Averaged), Iterations Aggregated`")
    out_lines.append("  - Inputs: condition means grouped by `(dataset, policy)` across rankers.")
    out_lines.append("  - Weighting: each ranker-condition contributes equally.")
    out_lines.append("- `Per-Policy (All Datasets, All Rankers Averaged), Iterations Aggregated`")
    out_lines.append("  - Inputs: condition means grouped by `policy` across dataset-ranker conditions.")
    out_lines.append("  - Weighting: each dataset-ranker condition contributes equally.")
    out_lines.append("- `Per-Policy (Pooled Across Datasets and Rankers), Iterations Aggregated`")
    out_lines.append("  - Inputs: all topic means pooled directly by `policy` (ignores condition boundaries).")
    out_lines.append("  - Weighting: each topic contributes equally globally; larger conditions get more total weight.")
    out_lines.append("- `Aggregated by Dataset x Ranker (Policies Averaged), Iterations Aggregated`")
    out_lines.append("  - Inputs: condition means grouped by `(dataset, ranker)` across policies.")
    out_lines.append("  - Weighting: each policy-condition contributes equally.")
    out_lines.append("")
    out_lines.append("### Notes")
    out_lines.append("")
    if args.precision_csv:
        out_lines.append("- Precision metrics are loaded from `--precision-csv` and override same-named log metrics.")
    out_lines.append("- Missing metrics are shown as `n/a`.")
    out_lines.append("")

    drp_stats = compute_stats(drp_topic_means, metrics)

    ranker_policy_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dataset_policy_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    policy_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    policy_topic_pooled_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dataset_ranker_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for (dataset, ranker, policy), metric_stats in drp_stats.items():
        for m in metrics:
            mean_val = metric_stats[m][0]
            if mean_val is None:
                continue
            ranker_policy_values[(ranker, policy)][m].append(mean_val)
            dataset_policy_values[(dataset, policy)][m].append(mean_val)
            policy_values[policy][m].append(mean_val)
            dataset_ranker_values[(dataset, ranker)][m].append(mean_val)
            # Pooled policy view: all topic-level values across datasets/rankers/logs.
            policy_topic_pooled_values[policy][m].extend(drp_topic_means[(dataset, ranker, policy)][m])
    # Summary table with bold maxima (iteration-aggregated across datasets).
    stats_summary = compute_stats(ranker_policy_values, metrics)
    out_lines.extend(
        render_table(
            "## Summary (Ranker × Policy, Iterations Aggregated; best per metric bolded)",
            ["Ranker", "Policy"],
            metrics,
            stats_summary,
            bold_max=True,
        )
    )

    # Dataset-aware tables (iteration aggregated).
    stats_dataset_ranker_policy = drp_stats
    out_lines.extend(
        render_table(
            "## Per-Condition (Dataset × Ranker × Policy), Iterations Aggregated",
            ["Dataset", "Ranker", "Policy"],
            metrics,
            stats_dataset_ranker_policy,
            bold_max=False,
        )
    )

    stats_dataset_policy = compute_stats(dataset_policy_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Condition (Dataset × Policy, Rankers Averaged), Iterations Aggregated",
            ["Dataset", "Policy"],
            metrics,
            stats_dataset_policy,
            bold_max=False,
        )
    )

    stats_policy = compute_stats(policy_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Policy (All Datasets, All Rankers Averaged), Iterations Aggregated",
            ["Policy"],
            metrics,
            stats_policy,
            bold_max=True,
        )
    )

    stats_policy_pooled = compute_stats(policy_topic_pooled_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Policy (Pooled Across Datasets and Rankers), Iterations Aggregated",
            ["Policy"],
            metrics,
            stats_policy_pooled,
            bold_max=True,
        )
    )

    stats_dataset_ranker = compute_stats(dataset_ranker_values, metrics)
    out_lines.extend(
        render_table(
            "## Aggregated by Dataset × Ranker (Policies Averaged), Iterations Aggregated",
            ["Dataset", "Ranker"],
            metrics,
            stats_dataset_ranker,
            bold_max=False,
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Saved summary to {output_path.resolve()}")


if __name__ == "__main__":
    main()
