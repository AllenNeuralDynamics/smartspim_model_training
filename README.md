# smartspim_cell_classifier

Benchmark a pre-trained SmartSPIM cell classification model and generate precision-recall plots.

## Usage

```python
from smartspim_cell_classifier.pipeline import SmartSPIMPipeline

pipeline = SmartSPIMPipeline.from_config("benchmark_config.yaml")
pipeline.run()
```

Or call stages individually:

```python
pipeline.benchmark()   # run inference, save CSVs and XMLs
pipeline.quantify()    # compute TP/FP/FN metrics, export to Excel
```

## Config

See `SmartSPIMPipeline.default_params()` for the full parameter schema.

Key fields:

| Field | Description |
|---|---|
| `trained_model_path` | Path to a local `.keras` model file |
| `benchmark_config_path` | Path to benchmark YAML (defines zarr paths, channels, regions) |
| `save_path` | Root directory for outputs |
| `benchmark_params.benchmark_json_path` | Directory containing ground-truth Neuroglancer JSON files |

## Outputs

- `classifications/<cond>_probabilities.csv` — per-cell probability scores
- `classifications/<cond>_classified_cells.xml` — classified cell locations
- `precision_recall_curves.png` — PR curve for each benchmarked condition
- `metrics.xlsx` — TP/FP/FN/Precision/Recall/F1 per condition

## Install

```bash
pip install -e .
```
