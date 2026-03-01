#!/usr/bin/env python
"""
Plot average-per-iteration cumulative precision metrics across topics.

Grid layout:
  rows = metrics (default: CumPrecision, CumPrecision@100)
  cols = rankers (default: bm25, splade)
Each cell shows per-policy grouped bars where each bar is the mean across
topics for that iteration. When multiple logs share the same ranker/policy,
their per-iteration means are averaged.

Example:
  python scripts/plot_cumprecision_avg_grid.py \\
    --log logs/final/dbpedia/oss120_bm25_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_bm25_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_bm25_fqkr_2.out --label forget_queries_keep_reason \\
    --log logs/final/dbpedia/oss120_splade_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_splade_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_splade_fqkr_0.out --label forget_queries_keep_reason \\
    --topicset beir/dbpedia-entity/test \\
    --offline-top100 always \\
    --ranking-model bm25 \\
    --index-name dbpedia_entity_bm25 \\
    --output analyses/grids/dbpedia_precision_avg_grid.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

try:
    import ir_datasets
except ImportError:  # pragma: no cover
    ir_datasets = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average-per-iteration cumulative precision metrics.")
    parser.add_argument("--log", action="append", required=True, help="Path to a JSONL log file.")
    parser.add_argument("--label", action="append", required=True, help="Label for each --log.")
    parser.add_argument(
        "--log-topicset",
        action="append",
        default=None,
        help="Per-log topicset aligned with --log order (optional; overrides --topicset).",
    )
    parser.add_argument(
        "--log-ranking-model",
        action="append",
        default=None,
        help="Per-log ranking model aligned with --log order (optional; overrides --ranking-model).",
    )
    parser.add_argument(
        "--log-index-name",
        action="append",
        default=None,
        help="Per-log index name aligned with --log order (optional; overrides --index-name).",
    )
    parser.add_argument(
        "--index-map",
        action="append",
        default=None,
        help=(
            "Index mapping in format '<topicset>|<ranker>|<index_name>'. "
            "Can be repeated. Used when --log-index-name/--index-name are not set."
        ),
    )
    parser.add_argument(
        "--log-encode-model",
        action="append",
        default=None,
        help="Per-log encode model aligned with --log order (optional; overrides --encode-model).",
    )
    parser.add_argument(
        "--log-device",
        action="append",
        default=None,
        help="Per-log device aligned with --log order (optional; overrides --device).",
    )
    parser.add_argument(
        "--metrics",
        default="CumPrecision,CumPrecision@100",
        help="Comma-separated metric keys (rows).",
    )
    parser.add_argument(
        "--ranker-order",
        default="bm25,splade",
        help="Comma-separated ranker order for columns (default: %(default)s).",
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
        help="Separator for suffix/prefix policy extraction.",
    )
    parser.add_argument(
        "--topicset",
        default=None,
        help="ir_datasets topicset. If omitted, inferred from logs.",
    )
    parser.add_argument("--y-max", type=float, default=0.0, help="Max value for y-axis (0 = auto per subplot).")
    parser.add_argument("--xtick-step", type=int, default=0, help="Show every Nth iteration tick (0 = auto).")
    parser.add_argument("--xtick-fontsize", type=int, default=8, help="X-axis tick font size.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--metrics-dir",
        default="",
        help=(
            "Directory to save computed precision metric files. "
            "Default: <output_stem>_metrics under output directory. "
            "Set to '-' to disable."
        ),
    )

    parser.add_argument(
        "--offline-top100",
        choices=["auto", "always", "never"],
        default="auto",
        help="How CumPrecision@100 is computed if missing from logs (default: auto).",
    )
    parser.add_argument("--ranking-model", default=None, help="Ranking model for offline top-100 reranking.")
    parser.add_argument("--index-name", default=None, help="OpenSearch index name for offline reranking.")
    parser.add_argument("--host", default="localhost", help="OpenSearch host.")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port.")
    parser.add_argument("--os-user", default=None, help="OpenSearch username (overrides env).")
    parser.add_argument("--os-pass", dest="os_pass", default=None, help="OpenSearch password (overrides env).")
    parser.add_argument("--no-auth", action="store_true", help="Disable OpenSearch basic auth.")
    parser.add_argument("--encode-model", default=None, help="Encoding model (needed for splade).")
    parser.add_argument("--device", default=None, help="Device hint for splade (cpu/cuda/mps).")
    parser.add_argument("--prf-docs", type=int, default=10, help="PRF docs for bm25_rm3.")
    parser.add_argument("--prf-terms", type=int, default=20, help="PRF terms for bm25_rm3.")
    parser.add_argument("--rerank-top-k", type=int, default=100, help="Top-k for rerank clients.")
    return parser.parse_args()


def auto_xtick_step(iterations: List[int]) -> int:
    if not iterations:
        return 1
    max_it = max(iterations)
    if max_it <= 15:
        return 1
    if max_it <= 30:
        return 2
    if max_it <= 60:
        return 4
    if max_it <= 100:
        return 5
    return 10


def is_cumulative_metric(metric_key: str) -> bool:
    key = metric_key.lower()
    return key.startswith("cum") or "cum" in key


def build_series(metric_key: str, pairs: List[Tuple[int, float]], max_iter: int) -> List[float | None]:
    values = {int(it): float(val) for it, val in pairs if val is not None}
    series: List[float | None] = []
    last: float | None = None
    cumulative = is_cumulative_metric(metric_key)
    for it in range(1, max_iter + 1):
        if it in values:
            last = values[it]
        if cumulative:
            series.append(last if last is not None else 0.0)
        else:
            series.append(values.get(it))
    return series


def normalize_rel_label(label_raw: object) -> Optional[bool]:
    if label_raw is None:
        return None
    s = str(label_raw).strip().upper()
    if s in {"NOTRELEVANT", "NOT_RELEVANT", "RELEVANCE.NOT_RELEVANT"} or s.endswith(".NOT_RELEVANT"):
        return False
    if s in {"RELEVANT", "RELEVANCE.RELEVANT"} or s.endswith(".RELEVANT"):
        return True
    return None


def derive_policy(label: str, mode: str, sep: str) -> str:
    known = [
        "forget_queries_keep_reason",
        "forget_queries_half",
        "forget_queries",
        "full",
    ]
    if mode == "auto":
        for key in known:
            if key in label:
                return key
    if mode == "label":
        return label
    parts = label.split(sep)
    if mode == "suffix":
        return parts[-1] if parts else label
    return parts[0] if parts else label


def resolve_metrics_dir(output_path: str, metrics_dir_arg: str) -> Optional[Path]:
    if metrics_dir_arg.strip() == "-":
        return None
    if metrics_dir_arg.strip():
        return Path(metrics_dir_arg).expanduser()
    out_path = Path(output_path)
    return out_path.parent / f"{out_path.stem}_metrics"


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class LogRunConfig:
    log_path: str
    label: str
    topicset: str
    ranking_model: Optional[str]
    index_name: Optional[str]
    encode_model: Optional[str]
    device: Optional[str]


def infer_log_metadata(log_path: str) -> tuple[Optional[str], Optional[str]]:
    inferred_dataset = None
    inferred_ranker = None
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("stage") != "ranking":
                continue
            if inferred_dataset is None and entry.get("dataset"):
                inferred_dataset = entry.get("dataset")
            if inferred_ranker is None and entry.get("ranker"):
                inferred_ranker = entry.get("ranker")
            if inferred_dataset is not None and inferred_ranker is not None:
                break
    return inferred_dataset, inferred_ranker


def resolve_per_log_values(values: Optional[List[str]], count: int, flag_name: str) -> List[Optional[str]]:
    if values is None:
        return [None] * count
    if len(values) != count:
        raise SystemExit(
            f"{flag_name} count ({len(values)}) must match --log count ({count}) when provided."
        )
    return [v if (v is not None and str(v).strip() != "") else None for v in values]


def resolve_per_log_values_sparse(
    values: Optional[List[str]],
    count: int,
    flag_name: str,
    applies_to: List[bool],
) -> List[Optional[str]]:
    if values is None:
        return [None] * count
    cleaned = [v if (v is not None and str(v).strip() != "") else None for v in values]
    if len(cleaned) == count:
        return cleaned
    required_count = sum(1 for a in applies_to if a)
    if len(cleaned) == required_count:
        out: List[Optional[str]] = [None] * count
        src_idx = 0
        for i, applies in enumerate(applies_to):
            if applies:
                out[i] = cleaned[src_idx]
                src_idx += 1
        return out
    raise SystemExit(
        f"{flag_name} count ({len(cleaned)}) must match --log count ({count}) "
        f"or the number of applicable logs ({required_count})."
    )


def needs_encode_model(ranking_model: Optional[str]) -> bool:
    return (ranking_model or "").lower() in {"splade", "bm25_dense", "bm25_cross_encoder", "dpr"}


def needs_device_hint(ranking_model: Optional[str]) -> bool:
    return (ranking_model or "").lower() == "splade"


def build_log_configs(args: argparse.Namespace) -> List[LogRunConfig]:
    num_logs = len(args.log)
    if len(args.label) != num_logs:
        raise SystemExit("Number of --log and --label arguments must match.")

    log_topicsets = resolve_per_log_values(args.log_topicset, num_logs, "--log-topicset")
    log_rankers = resolve_per_log_values(args.log_ranking_model, num_logs, "--log-ranking-model")
    log_indexes = resolve_per_log_values(args.log_index_name, num_logs, "--log-index-name")
    inferred_meta = [infer_log_metadata(lp) for lp in args.log]
    resolved_rankers = [
        log_rankers[i] or args.ranking_model or inferred_meta[i][1]
        for i in range(num_logs)
    ]
    encode_applies = [needs_encode_model(r) for r in resolved_rankers]
    device_applies = [needs_device_hint(r) for r in resolved_rankers]
    log_encodes = resolve_per_log_values_sparse(args.log_encode_model, num_logs, "--log-encode-model", encode_applies)
    log_devices = resolve_per_log_values_sparse(args.log_device, num_logs, "--log-device", device_applies)

    index_map: Dict[Tuple[str, str], str] = {}
    if args.index_map:
        for raw in args.index_map:
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) != 3:
                raise SystemExit(
                    f"Invalid --index-map '{raw}'. Expected '<topicset>|<ranker>|<index_name>'."
                )
            topicset_key, ranker_key, idx_name = parts
            if not topicset_key or not ranker_key or not idx_name:
                raise SystemExit(
                    f"Invalid --index-map '{raw}'. None of topicset/ranker/index can be empty."
                )
            index_map[(topicset_key, ranker_key.lower())] = idx_name

    configs: List[LogRunConfig] = []
    for idx, (log_path, label) in enumerate(zip(args.log, args.label)):
        inferred_dataset, inferred_ranker = inferred_meta[idx]

        topicset = log_topicsets[idx] or args.topicset or inferred_dataset
        if not topicset:
            raise SystemExit(
                f"Could not infer topicset for log '{log_path}'. Provide --topicset or --log-topicset."
            )

        ranking_model = log_rankers[idx] or args.ranking_model or inferred_ranker
        index_name = log_indexes[idx] or args.index_name
        if index_name is None and ranking_model is not None:
            index_name = index_map.get((topicset, ranking_model.lower()))
        encode_model = log_encodes[idx] or args.encode_model
        device = log_devices[idx] or args.device

        configs.append(
            LogRunConfig(
                log_path=log_path,
                label=label,
                topicset=topicset,
                ranking_model=ranking_model,
                index_name=index_name,
                encode_model=encode_model,
                device=device,
            )
        )
    return configs


def load_qrel_sets(topicset_name: str) -> Dict[str, set[str]]:
    if ir_datasets is None:
        raise SystemExit("ir_datasets is required to compute precision metrics.")
    ds = ir_datasets.load(topicset_name)
    qrels_by_topic: Dict[str, set[str]] = defaultdict(set)
    for row in ds.qrels_iter():
        if getattr(row, "relevance", 0) and row.relevance > 0:
            qrels_by_topic[row.query_id].add(row.doc_id)
    return qrels_by_topic


def create_offline_client(
    args: argparse.Namespace,
    dataset_name: str,
    ranking_model: str,
    index_name: str,
    encode_model: Optional[str],
    device: Optional[str],
):
    if args.no_auth:
        http_auth = None
    else:
        user = args.os_user or os.environ.get("OPENSEARCH_ADMIN_USER", "admin")
        password = args.os_pass or os.environ.get("OPENSEARCH_ADMIN_PASS", "admin")
        http_auth = (user, password)

    if ranking_model == "bm25":
        from geniie_lab.services.opensearch.opensearch_client_bm25 import OpenSearchClientBM25

        return OpenSearchClientBM25(
            index_name=index_name,
            dataset_name=dataset_name,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
        )
    if ranking_model == "splade":
        if not encode_model:
            raise SystemExit("--encode-model is required for --ranking-model splade.")
        from geniie_lab.services.opensearch.opensearch_client_splade import OpenSearchClientSplade

        return OpenSearchClientSplade(
            index_name=index_name,
            dataset_name=dataset_name,
            encode_model=encode_model,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
            device=device,
        )
    if ranking_model == "bm25_rm3":
        from geniie_lab.services.opensearch.opensearch_client_bm25_rm3 import OpenSearchClientBM25RM3

        return OpenSearchClientBM25RM3(
            index_name=index_name,
            dataset_name=dataset_name,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
            prf_docs=args.prf_docs,
            prf_terms=args.prf_terms,
        )
    if ranking_model == "bm25_dense":
        if not encode_model:
            raise SystemExit("--encode-model is required for --ranking-model bm25_dense.")
        from geniie_lab.services.opensearch.opensearch_client_bm25_dense import OpenSearchClientBM25Dense

        return OpenSearchClientBM25Dense(
            index_name=index_name,
            dataset_name=dataset_name,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
            encode_model=encode_model,
            rerank_top_k=args.rerank_top_k,
        )
    if ranking_model == "bm25_cross_encoder":
        if not encode_model:
            raise SystemExit("--encode-model is required for --ranking-model bm25_cross_encoder.")
        from geniie_lab.services.opensearch.opensearch_client_bm25_cross_encoder import OpenSearchClientBM25CrossEncoder

        return OpenSearchClientBM25CrossEncoder(
            index_name=index_name,
            dataset_name=dataset_name,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
            encode_model=encode_model,
            rerank_top_k=args.rerank_top_k,
        )
    if ranking_model == "dpr":
        if not encode_model:
            raise SystemExit("--encode-model is required for --ranking-model dpr.")
        from geniie_lab.services.opensearch.opensearch_client_dpr import OpenSearchClientDPR

        return OpenSearchClientDPR(
            index_name=index_name,
            dataset_name=dataset_name,
            host=args.host,
            port=args.port,
            http_auth=http_auth,
            encode_model=encode_model,
        )
    raise SystemExit(f"Unsupported ranking model for offline reranking: {ranking_model}")


def compute_avg_series(
    metric_key: str,
    per_topic_pairs: Dict[str, List[Tuple[int, float]]],
) -> tuple[List[int], List[float | None]]:
    if not per_topic_pairs:
        return [], []
    max_iter = max((it for pairs in per_topic_pairs.values() for it, _ in pairs), default=0)
    if max_iter <= 0:
        return [], []

    per_topic_series: List[List[float | None]] = []
    for pairs in per_topic_pairs.values():
        per_topic_series.append(build_series(metric_key, pairs, max_iter))

    avg_series: List[float | None] = []
    for idx in range(max_iter):
        vals = [s[idx] for s in per_topic_series if s[idx] is not None]
        avg_series.append((sum(vals) / len(vals)) if vals else None)

    return list(range(1, max_iter + 1)), avg_series


def load_log_data(
    args: argparse.Namespace,
    log_cfg: LogRunConfig,
    metrics: List[str],
    qrels_by_topic: Dict[str, set[str]],
) -> tuple[str, str, Dict[str, Dict[str, List[Tuple[int, float]]]]]:
    per_metric_topics: Dict[str, Dict[str, List[Tuple[int, float]]]] = {m: defaultdict(list) for m in metrics}
    ranking_counter: Dict[str, int] = defaultdict(int)
    last_query_by_topic: Dict[str, str] = {}
    judged_docids: Dict[str, set[str]] = defaultdict(set)
    judged_correct_rels: Dict[str, set[str]] = defaultdict(set)
    retrieved_top100_docids: Dict[str, set[str]] = defaultdict(set)
    retrieved_top100_rels: Dict[str, set[str]] = defaultdict(set)

    ranker = "unknown"
    dataset = "unknown"
    query_cache: Dict[str, List[str]] = {}

    # Determine whether we need offline support for this log.
    has_cumprecision100 = False
    if "CumPrecision@100" in metrics:
        with open(log_cfg.log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("stage") != "ranking":
                    continue
                perf = entry.get("performance") or {}
                if perf.get("CumPrecision@100") is not None:
                    has_cumprecision100 = True
                    break

    offline_client = None
    if "CumPrecision@100" in metrics:
        need_offline = (args.offline_top100 == "always") or (args.offline_top100 == "auto" and not has_cumprecision100)
        if need_offline:
            if not log_cfg.index_name:
                if args.offline_top100 == "always":
                    raise SystemExit(
                        f"--index-name/--log-index-name is required for offline top100 (log: {log_cfg.log_path})."
                    )
                print(
                    f"[WARN] {log_cfg.log_path}: CumPrecision@100 missing and no index configured; skipping offline.",
                    file=sys.stderr,
                )
            else:
                effective_ranker = log_cfg.ranking_model
                if effective_ranker is None:
                    if args.offline_top100 == "always":
                        raise SystemExit(
                            f"Cannot run offline top100 without ranker (log: {log_cfg.log_path}). "
                            "Set --ranking-model or --log-ranking-model."
                        )
                    print(
                        f"[WARN] {log_cfg.log_path}: could not infer ranker; skipping offline CumPrecision@100.",
                        file=sys.stderr,
                    )
                else:
                    offline_client = create_offline_client(
                        args=args,
                        dataset_name=log_cfg.topicset,
                        ranking_model=effective_ranker,
                        index_name=log_cfg.index_name,
                        encode_model=log_cfg.encode_model,
                        device=log_cfg.device,
                    )

    with open(log_cfg.log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            stage = entry.get("stage")
            topic = entry.get("topic_id", "<unknown>")

            if stage in ("query", "reformulation", "reformulate"):
                last_query_by_topic[topic] = entry.get("query", "")
                continue

            if stage == "rel_judge":
                docid = entry.get("docid")
                if docid:
                    judged_docids[topic].add(docid)
                    qrel_label = entry.get("qrel_label", 0) or 0
                    llm_rel = normalize_rel_label(entry.get("label"))
                    if qrel_label > 0 and llm_rel is True:
                        judged_correct_rels[topic].add(docid)
                continue

            if stage != "ranking":
                continue

            ranking_counter[topic] += 1
            it = ranking_counter[topic]
            ranker = entry.get("ranker") or ranker
            dataset = entry.get("dataset") or dataset

            if "CumPrecision" in metrics:
                denom = len(judged_docids[topic])
                num = len(judged_correct_rels[topic])
                value = (num / denom) if denom > 0 else 0.0
                per_metric_topics["CumPrecision"][topic].append((it, value))

            if "CumPrecision@100" in metrics:
                perf = entry.get("performance") or {}
                value = perf.get("CumPrecision@100")
                if value is None and offline_client is not None:
                    query_text = last_query_by_topic.get(topic, "")
                    if query_text:
                        if query_text not in query_cache:
                            try:
                                query_cache[query_text] = offline_client.search_docids(query_text, start=0, size=100)
                            except Exception as exc:
                                print(f"[WARN] Offline top100 retrieval failed: {exc}", file=sys.stderr)
                                query_cache[query_text] = []
                        top100_docids = query_cache[query_text]
                        rel_set = qrels_by_topic.get(topic, set())
                        for docid in top100_docids:
                            retrieved_top100_docids[topic].add(docid)
                            if docid in rel_set:
                                retrieved_top100_rels[topic].add(docid)
                        denom100 = len(retrieved_top100_docids[topic])
                        num100 = len(retrieved_top100_rels[topic])
                        value = (num100 / denom100) if denom100 > 0 else 0.0
                if value is not None:
                    try:
                        per_metric_topics["CumPrecision@100"][topic].append((it, float(value)))
                    except (TypeError, ValueError):
                        pass

    return ranker, dataset, per_metric_topics


def main() -> None:
    args = parse_args()
    log_configs = build_log_configs(args)
    metrics_dir = resolve_metrics_dir(args.output, args.metrics_dir)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    unsupported = [m for m in metrics if m not in {"CumPrecision", "CumPrecision@100"}]
    if unsupported:
        raise SystemExit(f"Unsupported metrics for this script: {unsupported}")

    ranker_order = [r.strip() for r in args.ranker_order.split(",") if r.strip()]
    if not ranker_order:
        raise SystemExit("No rankers specified.")

    qrels_cache: Dict[str, Dict[str, set[str]]] = {}

    # Collect per-log series first, then aggregate by (ranker, policy, metric).
    series_accumulator: Dict[Tuple[str, str, str], List[Tuple[List[int], List[float | None]]]] = defaultdict(list)
    datasets: set[str] = set()
    policy_order: List[str] = []
    per_topic_iteration_rows: List[Dict[str, object]] = []
    per_log_avg_series_rows: List[Dict[str, object]] = []
    aggregated_avg_series_rows: List[Dict[str, object]] = []

    for log_cfg in log_configs:
        policy = derive_policy(log_cfg.label, args.policy_label_from, args.policy_suffix_sep)
        if log_cfg.topicset not in qrels_cache:
            qrels_cache[log_cfg.topicset] = load_qrel_sets(log_cfg.topicset)
        ranker, dataset, per_metric_topics = load_log_data(
            args=args,
            log_cfg=log_cfg,
            metrics=metrics,
            qrels_by_topic=qrels_cache[log_cfg.topicset],
        )
        datasets.add(dataset)
        if policy not in policy_order:
            policy_order.append(policy)
        for metric_key, per_topic_pairs in per_metric_topics.items():
            for topic_id, pairs in per_topic_pairs.items():
                for iteration, value in pairs:
                    per_topic_iteration_rows.append(
                        {
                            "log_path": log_cfg.log_path,
                            "label": log_cfg.label,
                            "policy": policy,
                            "dataset": dataset,
                            "ranker": ranker,
                            "metric": metric_key,
                            "topic_id": topic_id,
                            "iteration": int(iteration),
                            "value": float(value),
                        }
                    )
            iterations, avg_series = compute_avg_series(metric_key, per_topic_pairs)
            if iterations:
                series_accumulator[(ranker, policy, metric_key)].append((iterations, avg_series))
                for iteration, value in zip(iterations, avg_series):
                    if value is None:
                        continue
                    per_log_avg_series_rows.append(
                        {
                            "log_path": log_cfg.log_path,
                            "label": log_cfg.label,
                            "policy": policy,
                            "dataset": dataset,
                            "ranker": ranker,
                            "metric": metric_key,
                            "iteration": int(iteration),
                            "value": float(value),
                        }
                    )

    # ranker -> policy -> metric -> (iterations, avg_series)
    series_map: Dict[str, Dict[str, Dict[str, Tuple[List[int], List[float | None]]]]] = defaultdict(lambda: defaultdict(dict))
    for (ranker, policy, metric_key), grouped_series in series_accumulator.items():
        max_iter = max((len(series) for _, series in grouped_series), default=0)
        if max_iter == 0:
            continue
        merged: List[float | None] = []
        for idx in range(max_iter):
            vals: List[float] = []
            for _, series in grouped_series:
                if idx < len(series) and series[idx] is not None:
                    vals.append(float(series[idx]))
            merged.append((sum(vals) / len(vals)) if vals else None)
        series_map[ranker][policy][metric_key] = (list(range(1, max_iter + 1)), merged)
        for iteration, value in enumerate(merged, start=1):
            if value is None:
                continue
            aggregated_avg_series_rows.append(
                {
                    "ranker": ranker,
                    "policy": policy,
                    "metric": metric_key,
                    "iteration": int(iteration),
                    "value": float(value),
                }
            )

    fig_rows = len(metrics)
    fig_cols = len(ranker_order)
    fig_width = max(10, fig_cols * 5.2)
    fig_height = max(6, fig_rows * 3.6)
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    if fig_rows == 1 and fig_cols == 1:
        axes_grid = [[axes]]
    elif fig_rows == 1:
        axes_grid = [axes]
    elif fig_cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    color_map = plt.colormaps.get_cmap("tab10")
    legend_handles = None
    legend_labels = None

    for r_idx, metric_key in enumerate(metrics):
        for c_idx, ranker in enumerate(ranker_order):
            ax = axes_grid[r_idx][c_idx]
            policies = series_map.get(ranker, {})
            max_iter = 0
            max_val = 0.0
            plotted = False

            for policy in policy_order:
                if policy not in policies:
                    continue
                iterations, avg_series = policies[policy].get(metric_key, ([], []))
                if not iterations:
                    continue
                max_iter = max(max_iter, max(iterations))
                for val in avg_series:
                    if val is not None and math.isfinite(val):
                        max_val = max(max_val, val)

            if max_iter == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{ranker} - {metric_key}")
                continue

            base_positions = list(range(max_iter))
            bar_width = 0.8 / max(1, len(policy_order))

            for p_idx, policy in enumerate(policy_order):
                if policy not in policies:
                    continue
                iterations, avg_series = policies[policy].get(metric_key, ([], []))
                if not iterations:
                    continue
                plotted = True
                heights: List[float] = []
                for idx in range(max_iter):
                    if idx < len(avg_series) and avg_series[idx] is not None:
                        heights.append(float(avg_series[idx]))
                    else:
                        heights.append(float("nan"))
                positions = [pos - 0.4 + bar_width / 2 + p_idx * bar_width for pos in base_positions]
                color = color_map(p_idx % color_map.N)
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    label=policy,
                )

            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{ranker} - {metric_key}")
                continue

            step = args.xtick_step or auto_xtick_step(list(range(1, max_iter + 1)))
            tick_positions = [pos for pos in base_positions[::step]]
            tick_labels = [str(i + 1) for i in base_positions[::step]]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis="x", labelsize=args.xtick_fontsize)
            ax.grid(alpha=0.3)
            ax.set_title(f"{ranker} - {metric_key}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric_key)
            if args.y_max > 0:
                ax.set_ylim(0, args.y_max)
            else:
                ax.set_ylim(0, (max_val * 1.08) if max_val > 0 else 1.0)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        legend_cols = max(1, min(4, len(legend_labels)))
        fig.legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.5, -0.01),
            loc="upper center",
            ncol=legend_cols,
            fontsize="small",
            title="Policy",
        )

    dataset_title = ", ".join(sorted(d for d in datasets if d and d != "unknown"))
    if dataset_title:
        fig.suptitle(f"Average per-iteration precision metrics across topics\nDataset(s): {dataset_title}", y=1.02)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {output_path.resolve()}")

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        per_topic_iteration_rows.sort(
            key=lambda r: (
                str(r["dataset"]),
                str(r["ranker"]),
                str(r["policy"]),
                str(r["metric"]),
                str(r["topic_id"]),
                int(r["iteration"]),
            )
        )
        per_log_avg_series_rows.sort(
            key=lambda r: (
                str(r["dataset"]),
                str(r["ranker"]),
                str(r["policy"]),
                str(r["metric"]),
                int(r["iteration"]),
                str(r["log_path"]),
            )
        )
        aggregated_avg_series_rows.sort(
            key=lambda r: (
                str(r["ranker"]),
                str(r["policy"]),
                str(r["metric"]),
                int(r["iteration"]),
            )
        )

        write_csv_rows(
            metrics_dir / "per_topic_iteration_values.csv",
            [
                "log_path",
                "label",
                "policy",
                "dataset",
                "ranker",
                "metric",
                "topic_id",
                "iteration",
                "value",
            ],
            per_topic_iteration_rows,
        )
        write_csv_rows(
            metrics_dir / "per_log_avg_series.csv",
            [
                "log_path",
                "label",
                "policy",
                "dataset",
                "ranker",
                "metric",
                "iteration",
                "value",
            ],
            per_log_avg_series_rows,
        )
        write_csv_rows(
            metrics_dir / "aggregated_avg_series.csv",
            [
                "ranker",
                "policy",
                "metric",
                "iteration",
                "value",
            ],
            aggregated_avg_series_rows,
        )

        readme_lines = [
            "# Precision Metric Exports",
            "",
            "This directory stores computed values produced by `plot_cumprecision_avg_grid.py`.",
            "",
            "- `per_topic_iteration_values.csv`: raw per-topic, per-iteration values reconstructed from logs.",
            "- `per_log_avg_series.csv`: per-log average across topics for each iteration.",
            "- `aggregated_avg_series.csv`: final averaged series used in the plotted bars.",
            "",
            "Formulas:",
            "",
            "- `CumPrecision_t = |judged_correct_rels_t| / |judged_docids_t|`",
            "- `CumPrecision@100_t = |retrieved_top100_rels_t| / |retrieved_top100_docids_t|`",
            "  (or logged `performance[\"CumPrecision@100\"]` when present)",
        ]
        (metrics_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        print(f"Saved metric exports to {metrics_dir.resolve()}")


if __name__ == "__main__":
    main()
