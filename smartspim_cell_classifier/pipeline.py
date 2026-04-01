import os
import time
import yaml
import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from glob import glob

from .utils import utils, classify, quantify, plots

logger = logging.getLogger(__name__)


class SmartSPIMPipeline:
    """
    SmartSPIM cell classification benchmark pipeline.

    Loads a pre-trained model, runs inference on zarr volumes, computes
    precision/recall metrics against ground truth annotations, and plots
    PR curves.

    Parameters
    ----------
    params : dict
        Top-level configuration dictionary (see from_config / default_params).
    """

    def __init__(self, params: dict):
        self.params = params

    def benchmark(self):
        """Run cell classification on zarr volumes."""
        p = self.params
        fname = p["benchmark_config_path"]

        for d in ("detections", "classifications", "plots"):
            os.makedirs(os.path.join(p["save_path"], d), exist_ok=True)

        with open(fname, "r") as f:
            data = yaml.safe_load(f)

        class_params = {
            **p["classification_params"],
            "model_path": p["trained_model_path"],
            "save_path": p["save_path"],
        }

        for cond, cond_data in data["data"].items():
            class_params["cells"] = quantify.get_all_cells(
                cond_data["zarrs"], cond_data["channel"], cond_data["region"], cond,
                p["save_path"], p["bucket_name"]
            )
            class_params["signal_path"] = f"{cond_data['zarrs']}{cond_data['channel']}.zarr"
            class_params["background_path"] = f"{cond_data['zarrs']}{cond_data['background']}.zarr"
            class_params["offsets"] = cond_data["region"]
            class_params["cond"] = cond

            start = time.time()
            classify.Classification(class_params).run()
            logger.info(f"Classification took {time.time() - start:.1f}s for condition: {cond}")

        plots_path = os.path.join(p["save_path"], "plots")
        self._plot_pr_curves(data, p, plots_path)

    def _plot_pr_curves(self, benchmark_data: dict, p: dict, plots_path: str) -> None:
        """Match saved probability CSVs against ground truth annotations and plot PR curves."""
        ann_path = p.get("benchmark_params", {}).get("benchmark_json_path")
        layer_name = p.get("benchmark_params", {}).get("layer_name", "cells")
        if not ann_path:
            logger.warning("benchmark_params.benchmark_json_path not set; skipping PR curve.")
            return

        dim_dict = {
            cond: [
                cd["region"][3] - cd["region"][0],
                cd["region"][4] - cd["region"][1],
                cd["region"][5] - cd["region"][2],
            ]
            for cond, cd in benchmark_data["data"].items()
        }

        annotations = {}
        for ap in glob(os.path.join(ann_path, "*.json")):
            info = os.path.basename(ap)[:-5].split("_")
            cond = "_".join(info[1:3])
            json_data = utils.load_json(ap)
            annotations[cond] = utils.parse_annotation_dict(json_data, layer_name, dim_dict[cond])

        pr_data = {}
        for cond in benchmark_data["data"]:
            prob_csv = os.path.join(p["save_path"], "classifications", f"{cond}_probabilities.csv")
            if not os.path.exists(prob_csv):
                logger.warning(f"No probability CSV found for {cond}, skipping.")
                continue
            if cond not in annotations:
                logger.warning(f"No annotations found for {cond}, skipping.")
                continue

            prob_df = pd.read_csv(prob_csv)
            det_locs = prob_df[["z", "y", "x"]].values
            y_score = prob_df["prob"].values

            ann_locs = np.array(annotations[cond])
            if len(ann_locs) == 0 or len(det_locs) == 0:
                logger.warning(f"Empty annotations or detections for {cond}, skipping.")
                continue

            tree = KDTree(ann_locs, leaf_size=2)
            matches = tree.query_radius(det_locs, r=7)
            y_true = np.array([1 if len(m) > 0 else 0 for m in matches])
            pr_data[cond] = (y_true, y_score)

        if pr_data:
            plots.run_all_plots(plots_path, pr_data=pr_data)

    def quantify(self):
        """Compute precision/recall metrics against benchmark annotations."""
        params = {
            **self.params["benchmark_params"],
            "classified_xml_path": os.path.join(self.params["save_path"], "classifications"),
            "save_path": self.params["save_path"],
            "benchmark_config_path": self.params["benchmark_config_path"],
        }
        quantify.run(params)

    def run(self):
        """Run benchmark then quantify."""
        if self.params.get("benchmark"):
            self.benchmark()
        if self.params.get("quantify"):
            self.quantify()

    @classmethod
    def from_config(cls, config_path: str) -> "SmartSPIMPipeline":
        """Instantiate the pipeline from a YAML config file."""
        with open(config_path) as f:
            params = yaml.safe_load(f)
        return cls(params)

    @staticmethod
    def default_params() -> dict:
        return {
            "benchmark": True,
            "quantify": False,
            "bucket_name": "aind-open-data",
            "save_path": "../results",
            "trained_model_path": "trained_models/model_01112026.keras",
            "benchmark_config_path": None,
            "classification_params": {
                "level": "1",
                "pad": 26,
                "batch_size": 32,
                "percentile_normalization": True,
                "percentile_range": (1, 99),
            },
            "benchmark_params": {
                "benchmark_json_path": "../data/benchmark_jsons/",
                "layer_name": "cells",
            },
        }
