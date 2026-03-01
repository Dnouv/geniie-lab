# Scripts Overview

This folder contains experiment runners, indexing utilities, and plotting tools.

## Indexing

Preferred entrypoint:

- `python scripts/index_preset.py --preset <touche2022|trec-covid> --retrieval <bm25|splade>`

Underlying indexers:

- `scripts/index_dataset.py` for BM25-style corpora
- `scripts/index_splade.py` for SPLADE-expanded corpora

Compatibility wrappers are still available:

- `scripts/index_touche_bm25.py`
- `scripts/index_touche_splade.py`
- `scripts/index_treccovid_bm25.py`
- `scripts/index_treccovid_splade.py`

## Session experiments

Session experiment scripts remain dataset-specific:

- `scripts/run_session_experiment.py`
- `scripts/run_session_experiment_splade.py`
- `scripts/run_session_experiment_esci.py`
- `scripts/run_session_experiment_touche_bm25.py`
- `scripts/run_session_experiment_touche_splade.py`
- `scripts/run_session_experiment_treccovid_bm25.py`
- `scripts/run_session_experiment_treccovid_splade.py`

## Plotting and summaries

Plot and analysis scripts include:

- `scripts/plot_*.py`
- `scripts/batch_bar_plots.py`
- `scripts/summarize_metrics_tables.py`
