# Scripts Guide

This directory contains indexing, experiment, plotting, and dataset utility scripts.

Use `.venv/bin/python` if you are running from this repo root and want the local environment:

- `.venv/bin/python scripts/<script_name>.py --help`

## Indexing scripts

- `index_preset.py`: preset-based entrypoint for common datasets (Touché 2022, TREC-COVID).
  - Example: `python scripts/index_preset.py --preset trec-covid --retrieval bm25 --recreate`
- `index_dataset.py`: generic BM25-style indexing from `ir_datasets` or Hugging Face corpora.
  - Example: `python scripts/index_dataset.py --dataset beir/dbpedia-entity --index dbpedia_entity_bm25 --recreate`
- `index_splade.py`: generic SPLADE indexing with term expansion.
  - Example: `python scripts/index_splade.py --dataset beir/dbpedia-entity --index dbpedia_entity_splade --device mps --recreate`
- `index_touche_bm25.py`: compatibility wrapper for Touché BM25 indexing.
  - Example: `python scripts/index_touche_bm25.py --recreate`
- `index_touche_splade.py`: compatibility wrapper for Touché SPLADE indexing.
  - Example: `python scripts/index_touche_splade.py --device mps --recreate`
- `index_treccovid_bm25.py`: compatibility wrapper for TREC-COVID BM25 indexing.
  - Example: `python scripts/index_treccovid_bm25.py --recreate`
- `index_treccovid_splade.py`: compatibility wrapper for TREC-COVID SPLADE indexing.
  - Example: `python scripts/index_treccovid_splade.py --device mps --recreate`

## Experiment runner scripts

- `run_session_experiment.py`: default DBPedia BM25 session experiment configuration.
  - Example: `python scripts/run_session_experiment.py > logs/session_dbpedia_bm25.out 2> logs/session_dbpedia_bm25.log`
- `run_session_experiment_splade.py`: DBPedia SPLADE session experiment.
  - Example: `python scripts/run_session_experiment_splade.py > logs/session_dbpedia_splade.out 2> logs/session_dbpedia_splade.log`
- `run_session_experiment_esci.py`: Amazon ESCI session experiment configuration.
  - Example: `python scripts/run_session_experiment_esci.py > logs/session_esci.out 2> logs/session_esci.log`
- `run_session_experiment_touche_bm25.py`: Touché 2022 BM25 session experiment.
  - Example: `python scripts/run_session_experiment_touche_bm25.py > logs/session_touche_bm25.out 2> logs/session_touche_bm25.log`
- `run_session_experiment_touche_splade.py`: Touché 2022 SPLADE session experiment.
  - Example: `python scripts/run_session_experiment_touche_splade.py > logs/session_touche_splade.out 2> logs/session_touche_splade.log`
- `run_session_experiment_treccovid_bm25.py`: TREC-COVID BM25 session experiment.
  - Example: `python scripts/run_session_experiment_treccovid_bm25.py > logs/session_treccovid_bm25.out 2> logs/session_treccovid_bm25.log`
- `run_session_experiment_treccovid_splade.py`: TREC-COVID SPLADE session experiment.
  - Example: `python scripts/run_session_experiment_treccovid_splade.py > logs/session_treccovid_splade.out 2> logs/session_treccovid_splade.log`
- `run_repetition_experiment.py`: repeated-stage experiment template.
  - Example: `python scripts/run_repetition_experiment.py`
- `run_agentic_experiment.py`: agentic loop experiment template (with next-action stage).
  - Example: `python scripts/run_agentic_experiment.py`
- `run_and_plot.py`: helper to run one experiment script and then plot results.
  - Example: `python scripts/run_and_plot.py --topicset beir/dbpedia-entity/test --run-id test1 --run-script scripts/run_session_experiment.py`

## Plotting and reporting scripts

- `plot_log_metrics.py`: general per-run charts (nDCG, RR, recall, query length, Jaccard, etc.).
  - Example: `python scripts/plot_log_metrics.py --log logs/session_dbpedia_bm25.out --topicset beir/dbpedia-entity/test --output-dir analyses/dbpedia_bm25`
- `plot_cumrecall_policies.py`: compare policy runs and generate per-topic segmented cumulative-recall plots.
  - Example: `python scripts/plot_cumrecall_policies.py --log logs/full.out --label full --log logs/fqh.out --label forget_queries_half --topicset beir/dbpedia-entity/test --output-dir analyses/cumrecall_policies`
- `plot_cumrecall_grid.py`: topic x metric grid with segmented policy bars.
  - Example: `python scripts/plot_cumrecall_grid.py --log logs/full.out --label full --log logs/fqh.out --label forget_queries_half --topicset beir/dbpedia-entity/test --output analyses/grids/cumrecall_grid.png`
- `plot_cumrecall_avg_grid.py`: average-per-iteration recall-style metric grid across topics.
  - Example: `python scripts/plot_cumrecall_avg_grid.py --log logs/bm25_full.out --label full --log logs/splade_full.out --label full --output analyses/grids/cumrecall_avg.png`
- `plot_cumprecision_avg_grid.py`: average-per-iteration precision-style metric grid across topics.
  - Example: `python scripts/plot_cumprecision_avg_grid.py --log logs/bm25_full.out --label full --topicset beir/dbpedia-entity/test --index-name dbpedia_entity_bm25 --output analyses/grids/cumprecision_avg.png`
- `plot_metric_groups_avg.py`: grouped avg metric grids (LLM metrics, retriever metrics, merged view).
  - Example: `python scripts/plot_metric_groups_avg.py --log logs/bm25_full.out --label full --log logs/splade_full.out --label full --output-dir analyses/metric_groups`
- `plot_image_grid.py`: tile existing images into one grid image.
  - Example: `python scripts/plot_image_grid.py --images analyses/a.png analyses/b.png --output analyses/grid.png`
- `batch_bar_plots.py`: run `plot_log_metrics` across many logs via glob patterns.
  - Example: `python scripts/batch_bar_plots.py --logs "logs/*.out" --output-root analyses --topicset beir/dbpedia-entity/test`
- `summarize_metrics_tables.py`: build markdown metric summary tables from multiple logs.
  - Example: `python scripts/summarize_metrics_tables.py --log logs/full.out --label full --log logs/fqh.out --label forget_queries_half --output analyses/summary/metrics.md`

## Dataset/topic inspection scripts

- `show_dataset_fields.py`: inspect document/query/qrel schema for datasets.
  - Example: `python scripts/show_dataset_fields.py --dataset beir/dbpedia-entity`
- `show_selected_topics.py`: display topic text for selected topic IDs.
  - Example: `python scripts/show_selected_topics.py --topicset beir/dbpedia-entity/test --topic-id INEX_LD-2010069`
- `show_topic_prefix_stats.py`: inspect topic prefix/statistics patterns.
  - Example: `python scripts/show_topic_prefix_stats.py --topicset beir/dbpedia-entity/test`
- `show_topic_rels.py`: inspect relevance statistics per topic.
  - Example: `python scripts/show_topic_rels.py --topicset beir/dbpedia-entity/test`
- `analyze_context_trends.py`: analyze context/trace trends from experiment outputs.
  - Example: `python scripts/analyze_context_trends.py --help`

## Notes

- Most experiment scripts are configuration-driven templates; edit settings inside the script when needed.
- For long runs, always redirect stdout/stderr to log files as shown above.
- Use `--help` on each script to view all supported flags and defaults.
