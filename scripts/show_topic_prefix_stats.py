#!/usr/bin/env python
"""
Summarize qrels coverage by topic prefix for one or more ir_datasets topicsets.

Examples:
  python scripts/show_topic_prefix_stats.py --topicset beir/dbpedia-entity
  python scripts/show_topic_prefix_stats.py --topicset beir/dbpedia-entity/test --csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import ir_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show qrels coverage by topic prefix.")
    parser.add_argument(
        "--topicset",
        required=True,
        help="Dataset identifier (e.g., beir/dbpedia-entity or beir/dbpedia-entity/test).",
    )
    parser.add_argument(
        "--thresholds",
        default="10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated thresholds for >= relevant-doc counts (default: %(default)s).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output CSV instead of a table.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Show up to N sample queries per prefix (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=3,
        help="Show up to N relevant docids per sample query (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-doc-text",
        action="store_true",
        help="Include document title/text previews for sample relevant docs (may be slow).",
    )
    parser.add_argument(
        "--doc-preview-chars",
        type=int,
        default=200,
        help="Max characters for document preview text (default: %(default)s).",
    )
    return parser.parse_args()


def topic_prefix(qid: str) -> str:
    if qid.startswith("QALD2_te-"):
        return "QALD2_te"
    if qid.startswith("QALD2_tr-"):
        return "QALD2_tr"
    if qid.startswith("INEX_LD-"):
        return "INEX_LD"
    if qid.startswith("INEX_XER-"):
        return "INEX_XER"
    if qid.startswith("SemSearch_E-"):
        return "SemSearch_E"
    if qid.startswith("SemSearch_LS-"):
        return "SemSearch_LS"
    if qid.startswith("TREC_Entity-"):
        return "TREC_Entity"
    return "OTHER"


def category_label(base_id: str, dataset_id: str) -> str:
    if dataset_id == base_id:
        return "base"
    prefix = f"{base_id}/"
    if dataset_id.startswith(prefix):
        return dataset_id[len(prefix):]
    return dataset_id


def parse_thresholds(raw: str) -> List[int]:
    values: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    if not values:
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    return sorted(set(values))


def topic_text(topic) -> str:
    for field in ("title", "text", "query", "description", "narrative"):
        value = getattr(topic, field, None)
        if value:
            return str(value)
    return ""


def doc_title_text(doc) -> tuple[str, str]:
    title_fields = ("title", "source_title", "topic")
    text_fields = ("text", "premises_texts", "conclusion", "source_text")

    title = ""
    for field in title_fields:
        value = getattr(doc, field, None)
        if value:
            title = str(value)
            break

    text_val = ""
    for field in text_fields:
        value = getattr(doc, field, None)
        if value:
            text_val = str(value)
            break

    return title, text_val


def resolve_topicsets(topicset: str) -> Tuple[str, List[str]]:
    try:
        base_id = ir_datasets.queries_parent_id(topicset)
    except Exception:
        base_id = topicset
    candidates = [
        name
        for name in ir_datasets.registry
        if name == base_id or name.startswith(f"{base_id}/")
    ]
    topicsets = []
    for name in sorted(candidates):
        dataset = ir_datasets.load(name)
        if dataset.has("queries") and dataset.has("qrels"):
            topicsets.append(name)
    if not topicsets:
        topicsets = [topicset]
    return base_id, topicsets


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    base_id, topicsets = resolve_topicsets(args.topicset)

    sample_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    sample_docs: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    sample_doc_ids: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dataset_map: Dict[str, object] = {}
    if not args.sample_doc_text and args.doc_preview_chars != 200:
        args.sample_doc_text = True
        print("Note: --doc-preview-chars set; enabling --sample-doc-text.")

    if not args.csv:
        categories = [category_label(base_id, name) for name in topicsets]
        print(f"Base dataset: {base_id}")
        print("Available categories: " + ", ".join(categories))
        print()

    rows: List[Dict[str, object]] = []
    for topicset in topicsets:
        dataset = ir_datasets.load(topicset)
        category = category_label(base_id, topicset)
        dataset_map[category] = dataset

        # Map topic IDs to prefixes and capture samples.
        topics_by_prefix: Dict[str, List[str]] = defaultdict(list)
        sample_qids: set[str] = set()
        for topic in dataset.queries_iter():
            prefix_name = topic_prefix(topic.query_id)
            topics_by_prefix[prefix_name].append(topic.query_id)
            if args.samples > 0 and len(sample_map[category][prefix_name]) < args.samples:
                sample_map[category][prefix_name].append(
                    (topic.query_id, topic_text(topic))
                )
                sample_qids.add(topic.query_id)

        # Count relevant docs per topic (relevance > 0) and capture sample docids.
        rel_counts: Dict[str, int] = defaultdict(int)
        for qrel in dataset.qrels_iter():
            if getattr(qrel, "relevance", 0) and qrel.relevance > 0:
                rel_counts[qrel.query_id] += 1
                if args.samples > 0 and qrel.query_id in sample_qids:
                    if len(sample_docs[category][qrel.query_id]) < args.sample_docs:
                        if qrel.doc_id not in sample_doc_ids[category][qrel.query_id]:
                            sample_docs[category][qrel.query_id].append((qrel.doc_id, qrel.relevance))
                            sample_doc_ids[category][qrel.query_id].add(qrel.doc_id)

        for prefix_name, topic_ids in topics_by_prefix.items():
            counts = [rel_counts.get(tid, 0) for tid in topic_ids]
            topics_total = len(topic_ids)
            topics_with_rels = sum(1 for c in counts if c > 0)
            total_rels = sum(counts)
            max_rels = max(counts) if counts else 0
            avg_rels = round(total_rels / topics_with_rels, 1) if topics_with_rels else 0.0

            row: Dict[str, object] = {
                "base_dataset": base_id,
                "category": category_label(base_id, topicset),
                "prefix": prefix_name,
                "topics": topics_total,
                "with_rels": topics_with_rels,
                "total_rels": total_rels,
                "max_rels": max_rels,
                "avg_rels": avg_rels,
            }
            for threshold in thresholds:
                row[f">={threshold}"] = sum(1 for c in counts if c >= threshold)
            rows.append(row)

    rows.sort(key=lambda x: (str(x["category"]), str(x["prefix"])))

    threshold_cols = [f">={t}" for t in thresholds]

    if args.csv:
        header = ["base_dataset", "category", "prefix", "topics", "with_rels"] + threshold_cols + ["total_rels", "max_rels", "avg_rels"]
        print(",".join(header))
        for row in rows:
            print(",".join(str(row[col]) for col in header))
        return

    col_names = ["base_dataset", "category", "prefix", "topics", "with_rels"] + threshold_cols + ["total_rels", "max_rels", "avg_rels"]
    col_widths: Dict[str, int] = {}
    for name in col_names:
        col_widths[name] = len(name)
    for row in rows:
        for name in col_names:
            col_widths[name] = max(col_widths[name], len(str(row[name])))

    header = "  ".join(name.ljust(col_widths[name]) for name in col_names)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[name]).ljust(col_widths[name]) for name in col_names))

    if args.samples > 0:
        print()
        print("Sample queries + relevant docids:")
        for category in sorted(sample_map.keys()):
            print(f"\n[{category}]")
            docstore = None
            if args.sample_doc_text:
                try:
                    docstore = dataset_map[category].docs_store()
                except Exception as exc:
                    print(f"  (docstore unavailable: {exc})")
            for prefix_name in sorted(sample_map[category].keys()):
                print(f"  {prefix_name}:")
                for qid, text in sample_map[category][prefix_name]:
                    preview = text.replace("\n", " ").strip()
                    print(f"    - {qid}: {preview}")
                    rel_list = sample_docs.get(category, {}).get(qid, [])
                    if rel_list:
                        rel_preview = ", ".join(f"{docid}({rel})" for docid, rel in rel_list)
                        print(f"      rel_docs: {rel_preview}")
                        if args.sample_doc_text and docstore is not None:
                            for docid, rel in rel_list:
                                parts = [p.strip() for p in docid.split(",") if p.strip()] or [docid]
                                for idx, part in enumerate(parts, start=1):
                                    lookup_id = part
                                    try:
                                        doc = docstore.get(lookup_id)
                                    except Exception:
                                        doc = None
                                    if not doc and "__" in part:
                                        lookup_id = part.split("__", 1)[0]
                                        try:
                                            doc = docstore.get(lookup_id)
                                        except Exception:
                                            doc = None
                                    if not doc:
                                        print(f"        doc_{idx}_id: {part} (not found)")
                                        continue
                                    title, text_val = doc_title_text(doc)
                                    text_val = text_val.replace("\n", " ").strip()
                                    if args.doc_preview_chars and len(text_val) > args.doc_preview_chars:
                                        text_val = text_val[: args.doc_preview_chars] + "..."
                                    label = f"{part}"
                                    if lookup_id != part:
                                        label = f"{part} -> {lookup_id}"
                                    print(f"        doc_{idx}_id: {label} (rel={rel})")
                                    if title:
                                        print(f"        doc_{idx}_title: {title}")
                                    if text_val:
                                        print(f"        doc_{idx}_text: {text_val}")
                    else:
                        print("      rel_docs: none")


if __name__ == "__main__":
    main()
