#!/usr/bin/env python
"""
Summarize ranking metrics from JSONL experiment logs into markdown tables.

Outputs:
1) Per-condition table (ranker x policy) with mean/std across topics.
2) Overall table aggregated by ranker (mean/std across policies and topics).

Example:
  python scripts/summarize_metrics_tables.py \\
    --log logs/final/dbpedia/oss120_bm25_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_bm25_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_bm25_fqkr_2.out --label forget_queries_keep_reason \\
    --log logs/final/dbpedia/oss120_splade_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_splade_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_splade_fqkr_0.out --label forget_queries_keep_reason \\
    --metrics "CumRecall,CumRecall@100,nDCG@10,RR@10,R@100" \\
    --output analyses/summary/metrics_tables.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def load_iteration_metrics(
    log_path: str,
    metrics: List[str],
) -> tuple[str, str, Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
    per_topic_iters: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    last_per_topic: Dict[str, Dict[str, float]] = {}
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
            last_per_topic[topic] = {m: perf.get(m) for m in metrics}
            for m in metrics:
                value = perf.get(m)
                if value is None:
                    continue
                try:
                    per_topic_iters[topic][m].append(float(value))
                except (TypeError, ValueError):
                    continue

    return ranker, dataset, per_topic_iters, last_per_topic


def format_mean_std(mean: float, std: float) -> str:
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
            max_by_metric[m] = max((stats[key][m][0] for key in stats), default=0.0)

    for raw_key in sorted(stats.keys()):
        key_tuple = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        row = list(key_tuple)
        for m in metrics:
            mean, std = stats[raw_key][m]
            cell = format_mean_std(mean, std)
            if bold_max and mean == max_by_metric.get(m, None):
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

    # condition -> metric -> list of per-topic values
    condition_values: Dict[tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    condition_iter_values: Dict[tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    condition_values_agg: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    condition_iter_values_agg: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ranker_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dataset_ranker_iter_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dataset_policy_iter_values: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    policy_iter_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

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
            return parts[0] if parts else label
        return parts[-1] if parts else label

    for log_path, label in zip(args.log, args.label):
        ranker, dataset, per_topic_iters, per_topic = load_iteration_metrics(log_path, metrics)
        policy = derive_policy(label)
        condition_key = (dataset, ranker, policy)
        condition_key_agg = (ranker, policy)
        for topic_id, metric_map in per_topic.items():
            for m in metrics:
                value = metric_map.get(m)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                condition_values[condition_key][m].append(value_f)
                condition_values_agg[condition_key_agg][m].append(value_f)
                ranker_values[(dataset, ranker)][m].append(value_f)
        for topic_id, metric_iters in per_topic_iters.items():
            for m, values in metric_iters.items():
                if not values:
                    continue
                mean_val, _ = mean_std(values)
                condition_iter_values[condition_key][m].append(mean_val)
                condition_iter_values_agg[condition_key_agg][m].append(mean_val)
                dataset_ranker_iter_values[(dataset, ranker)][m].append(mean_val)
                dataset_policy_iter_values[(dataset, policy)][m].append(mean_val)
                policy_iter_values[policy][m].append(mean_val)

    # Build markdown
    out_lines: List[str] = []
    out_lines.append("# Metric Summary Tables")
    out_lines.append("")
    # Summary table with bold maxima (iteration-aggregated across datasets).
    stats_summary = compute_stats(condition_iter_values_agg, metrics)
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
    stats_dataset_ranker_policy = compute_stats(condition_iter_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Condition (Dataset × Ranker × Policy), Iterations Aggregated",
            ["Dataset", "Ranker", "Policy"],
            metrics,
            stats_dataset_ranker_policy,
            bold_max=False,
        )
    )

    stats_dataset_policy = compute_stats(dataset_policy_iter_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Condition (Dataset × Policy, Rankers Pooled), Iterations Aggregated",
            ["Dataset", "Policy"],
            metrics,
            stats_dataset_policy,
            bold_max=False,
        )
    )

    stats_policy = compute_stats(policy_iter_values, metrics)
    out_lines.extend(
        render_table(
            "## Per-Policy (All Datasets, All Rankers), Iterations Aggregated",
            ["Policy"],
            metrics,
            stats_policy,
            bold_max=True,
        )
    )

    stats_dataset_ranker = compute_stats(dataset_ranker_iter_values, metrics)
    out_lines.extend(
        render_table(
            "## Aggregated by Dataset × Ranker (All Policies), Iterations Aggregated",
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
