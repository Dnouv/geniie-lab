#!/usr/bin/env python
"""
Create a grid image from multiple plot images (PNG/JPG).

Examples:
  python scripts/plot_image_grid.py \\
    --glob "analyses/cumrecall_policies_dbpedia/**/by_topic/*.png" \\
    --max 12 \\
    --output analyses/grids/dbpedia_by_topic_grid.png

  python scripts/plot_image_grid.py \\
    --input analyses/a.png --input analyses/b.png --input analyses/c.png \\
    --rows 2 --cols 2 \\
    --labels-from-filenames \\
    --output analyses/grids/custom_grid.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import fnmatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile multiple images into a grid.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Path to an image file (repeatable).",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern for input images (repeatable).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of images to include (0 = no limit).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="Number of grid rows (0 = auto).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Number of grid columns (0 = auto).",
    )
    parser.add_argument(
        "--labels-from-filenames",
        action="store_true",
        help="Use filenames as small titles for each cell.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude images whose path contains this substring (repeatable).",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude images whose path matches this glob (repeatable).",
    )
    parser.add_argument(
        "--grid-metrics",
        default="",
        help="Comma-separated metric identifiers to stack vertically (e.g., CumRecall,CumRecall_100,Jaccard).",
    )
    parser.add_argument(
        "--grid-metric-in-path",
        default="",
        help="Comma-separated path substrings that identify each metric (must align with --grid-metrics).",
    )
    parser.add_argument(
        "--grid-topics",
        default="",
        help="Comma-separated topic IDs to order across columns.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI (default: %(default)s).",
    )
    parser.add_argument(
        "--cell-scale",
        type=float,
        default=1.0,
        help="Scale each cell size relative to source image pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path.",
    )
    return parser.parse_args()


def collect_images(paths: List[str], globs: List[str]) -> List[Path]:
    items: List[Path] = []
    for p in paths:
        items.append(Path(p))
    for pattern in globs:
        items.extend(sorted(Path().glob(pattern)))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def build_topic_metric_grid(
    images: List[Path],
    topics: List[str],
    metrics: List[str],
    metric_markers: List[str],
) -> List[Path]:
    by_metric: List[List[Path]] = []
    for marker in metric_markers:
        matched = [p for p in images if marker in str(p)]
        by_metric.append(matched)

    grid: List[Path] = []
    for metric_idx in range(len(metrics)):
        pool = by_metric[metric_idx]
        for topic in topics:
            hit = next((p for p in pool if topic in p.name), None)
            if hit is None:
                hit = next((p for p in pool if topic in str(p)), None)
            if hit is None:
                hit = Path("")
            grid.append(hit)
    return grid


def resolve_grid(n: int, rows: int, cols: int) -> tuple[int, int]:
    if rows > 0 and cols > 0:
        return rows, cols
    if rows > 0:
        cols = math.ceil(n / rows)
        return rows, cols
    if cols > 0:
        rows = math.ceil(n / cols)
        return rows, cols
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols) if cols else 1
    return rows, cols


def main() -> None:
    args = parse_args()
    images = collect_images(args.input, args.glob)
    if args.exclude or args.exclude_glob:
        filtered: List[Path] = []
        for path in images:
            path_str = str(path)
            if any(ex in path_str for ex in args.exclude):
                continue
            if any(fnmatch.fnmatch(path_str, pattern) for pattern in args.exclude_glob):
                continue
            filtered.append(path)
        images = filtered
    if args.max > 0:
        images = images[: args.max]
    if not images:
        raise SystemExit("No images found. Provide --input or --glob.")

    grid_topics = [t.strip() for t in args.grid_topics.split(",") if t.strip()]
    grid_metrics = [m.strip() for m in args.grid_metrics.split(",") if m.strip()]
    grid_metric_markers = [m.strip() for m in args.grid_metric_in_path.split(",") if m.strip()]
    use_structured_grid = bool(grid_topics and grid_metrics and grid_metric_markers)
    if use_structured_grid and len(grid_metrics) != len(grid_metric_markers):
        raise SystemExit("--grid-metrics and --grid-metric-in-path must have the same length.")

    if use_structured_grid:
        images = build_topic_metric_grid(images, grid_topics, grid_metrics, grid_metric_markers)

    rows, cols = resolve_grid(len(images), args.rows, args.cols)

    loaded = []
    for path in images:
        if not path or not path.exists():
            loaded.append(None)
        else:
            loaded.append(plt.imread(path))
    heights = [img.shape[0] for img in loaded if img is not None]
    widths = [img.shape[1] for img in loaded if img is not None]
    cell_h = max(heights) if heights else 300
    cell_w = max(widths) if widths else 400

    fig_width = (cols * cell_w * args.cell_scale) / args.dpi
    fig_height = (rows * cell_h * args.cell_scale) / args.dpi

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=args.dpi)
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel()
    else:
        axes_list = [axes]

    for ax in axes_list[len(images):]:
        ax.axis("off")

    for idx, path in enumerate(images):
        ax = axes_list[idx]
        img = loaded[idx]
        if img is None:
            ax.axis("off")
            if use_structured_grid:
                row = idx // cols
                col = idx % cols
                if row < len(grid_metrics) and col < len(grid_topics):
                    ax.text(
                        0.5,
                        0.5,
                        "missing",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            continue
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        if args.labels_from_filenames:
            ax.set_title(path.stem, fontsize=8)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid to {output_path.resolve()}")


if __name__ == "__main__":
    main()
