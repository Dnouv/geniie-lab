#!/usr/bin/env python
"""
Show available fields for docs / queries / qrels in an ir_datasets dataset.

Example:
  python scripts/show_dataset_fields.py --dataset argsme/2020-04-01/processed/touche-2022-task-1
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, Optional

import ir_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show field names for docs/queries/qrels.")
    parser.add_argument("--dataset", required=True, help="Dataset identifier.")
    parser.add_argument("--preview-chars", type=int, default=200, help="Max chars for string previews.")
    return parser.parse_args()


def _shorten(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    if limit and len(text) > limit:
        return text[:limit] + "..."
    return text


def _value_preview(value: Any, limit: int) -> str:
    if isinstance(value, str):
        return _shorten(value, limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        return f"list(len={len(value)}): {str(value[0])[:limit]}..."
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:5]})"
    return str(value)


def _record_fields(record: Any) -> Dict[str, Any]:
    if record is None:
        return {}
    if hasattr(record, "_asdict"):
        try:
            return record._asdict()
        except Exception:
            pass
    if hasattr(record, "_fields"):
        return {field: getattr(record, field) for field in record._fields}
    fields = {}
    for name in dir(record):
        if name.startswith("_"):
            continue
        try:
            value = getattr(record, name)
        except Exception:
            continue
        if callable(value):
            continue
        fields[name] = value
    return fields


def _print_record(title: str, record: Optional[Any], preview_chars: int) -> None:
    print(f"\n{title}:")
    if record is None:
        print("  (none)")
        return
    print(f"  type: {type(record).__name__}")
    fields = _record_fields(record)
    if not fields:
        print("  (no fields detected)")
        return
    for key in sorted(fields.keys()):
        value = fields[key]
        print(f"  - {key}: {_value_preview(value, preview_chars)}")


def main() -> None:
    args = parse_args()
    dataset = ir_datasets.load(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Has docs: {dataset.has('docs')}")
    print(f"Has queries: {dataset.has('queries')}")
    print(f"Has qrels: {dataset.has('qrels')}")

    if dataset.has("docs"):
        doc = next(dataset.docs_iter(), None)
        _print_record("Doc fields", doc, args.preview_chars)

    if dataset.has("queries"):
        query = next(dataset.queries_iter(), None)
        _print_record("Query fields", query, args.preview_chars)

    if dataset.has("qrels"):
        qrel = next(dataset.qrels_iter(), None)
        _print_record("Qrel fields", qrel, args.preview_chars)


if __name__ == "__main__":
    main()
