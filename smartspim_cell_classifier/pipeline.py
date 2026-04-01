import os
import time
import yaml
import logging
import s3fs

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from glob import glob
from pathlib import Path
from imlib.IO.cells import get_cells

from .utils import utils, benchmark_keras, quantify, plots
from .train import train_keras

logger = logging.getLogger(__name__)

# Fields that must be explicitly set by the user (not null) when their
# parent stage is enabled.
_REQUIRED_BY_STAGE = {
    "train_model":        ["save_path", "trained_model_path", "yml_path", "training_sets"],
    "build_training_set": ["save_path", ("annotation_params", "annotation_path")],
    "benchmark":          ["save_path", "trained_model_path", "benchmark_config_path"],
    "quantify":           [("benchmark_params", "classified_xml_path"),
                           ("benchmark_params", "benchmark_json_path")],
}


def _check_required(params: dict) -> None:
    """Raise ValueError listing every required field that is still null."""
    missing = []
    for stage, fields in _REQUIRED_BY_STAGE.items():
        if not params.get(stage):
            continue
        for field in fields:
            if isinstance(field, tuple):
                section, key = field
                if params.get(section, {}).get(key) is None:
                    missing.append(f"  {section}.{key}  (needed for {stage})")
            else:
                if params.get(field) is None:
                    missing.append(f"  {field}  (needed for {stage})")
    if missing:
        raise ValueError(
            "The following required config fields are not set:\n"
            + "\n".join(missing)
            + "\n\nEdit your config YAML or pass values explicitly."
        )


def _compile_ymls(yml_path, training_sets):
    yml_files = []
    if str(yml_path).startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        for training_set in training_sets:
            pattern = f"{str(yml_path).rstrip('/')}/{training_set}/ymls/*.yml"
            yml_files.extend("s3://" + f for f in fs.glob(pattern))
    else:
        for training_set in training_sets:
            files = glob(os.path.join(yml_path, training_set, "ymls", "*.yml"))
            yml_files.extend(Path(f) for f in files)
    return yml_files


def _training_configs(yml_path, training_sets, model_path, lr, schedule, monitor):
    return {
        "output_dir": Path("../results"),
        "yaml_file": _compile_ymls(yml_path, training_sets),
        "trained_model": Path(model_path),
        "model_weights": Path(model_path),
        "model": os.path.basename(model_path),
        "learning_rate": lr,
        "continue_training": True,
        "test_fraction": 0.1,
        "batch_size": 128,
        "save_progress": False,
        "epochs": 150,
        "schedule": schedule,
        "monitor": monitor,
    }


class SmartSPIMPipeline:
    """
    Orchestrates the SmartSPIM cell classification pipeline.

    Stages are independent methods that can be called individually or
    sequentially via run(). The trained Keras model is stored as an
    instance variable so it flows naturally from build_model -> train_model.

    Parameters
    ----------
    params : dict
        Top-level configuration dictionary (see default_params() for schema).
    """

    def __init__(self, params: dict):
        _check_required(params)
        self.params = params
        self.model = None
        self.model_name = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def build_model(self):
        """Build and compile the ResNet model."""
        logger.info("Building ResNet model")
        self.model, self.model_name = utils.build_model(
            save_path=self.params["save_path"],
            model_params=self.params["model_params"],
        )

    def build_training_set(self):
        """Fetch zarrs from AWS and save image patches for training."""
        utils.build_training_set(
            save_path=self.params["save_path"],
            **self.params["annotation_params"],
        )

    def train_model(self):
        """Train the model using the compiled yml dataset."""
        if self.model is None:
            raise ValueError(
                "No model available. Call build_model() before train_model()."
            )

        p = self.params
        data_params = _training_configs(
            p["yml_path"],
            p["training_sets"],
            p["trained_model_path"],
            p["model_params"]["learning_rate"],
            p["schedule"],
            p["monitor"],
        )
        data_params["balance"] = p["balance"]
        data_params["override_cube_dir"] = p.get("override_cube_dir", False)

        logger.info(f"Augmentation parameters: {p['augment_params']}")
        train_keras.run(data_params, p["augment_params"], self.model)

    def benchmark(self):
        """Run cell classification on zarr volumes."""
        p = self.params
        fname = p["benchmark_config_path"]

        for d in ("detections", "classifications", "scaled_cells", "crops"):
            os.makedirs(f"../results/{d}/", exist_ok=True)

        with open(fname, "r") as f:
            data = yaml.safe_load(f)

        aug = p["augment_params"]
        class_params = {
            "model": p["trained_model_path"],
            "save_path": p["save_path"],
            "network_depth": "18-layer",
            "level": "1",
            "detect_pad": 20,
            "pad": 26,
            "test": p["Test"],
            "batch_size": 32,
            "rescale": None,
            "percentile_normalization": aug.get("percentile_normalization", False),
            "percentile_range": aug.get("percentile_range", (1, 99)),
        }

        for cond, cond_data in data["data"].items():
            if p["Test"]:
                class_params["cells"] = quantify.get_all_cells(
                    cond_data["zarrs"],
                    cond_data["channel"],
                    data["data"][cond]["region"],
                    cond,
                )
            else:
                class_params["cells"] = get_cells(
                    glob(os.path.join("../data/camilo_detected/", f"{cond}*"))[0]
                )

            class_params["signal_path"] = (
                f"{cond_data['zarrs']}{cond_data['channel']}.zarr"
            )
            class_params["background_path"] = (
                f"{cond_data['zarrs']}{cond_data['background']}.zarr"
            )
            class_params["offsets"] = cond_data["region"]
            class_params["cond"] = cond

            start = time.time()
            benchmark_keras.Classification(class_params).run()
            logger.info(f"Classification took {time.time() - start:.1f}s for condition: {cond}")

        self._plot_pr_curves(data, p)

    def _plot_pr_curves(self, benchmark_data: dict, p: dict) -> None:
        """Match saved probability CSVs against ground truth annotations and plot PR curves."""
        ann_path = p.get("benchmark_params", {}).get("benchmark_json_path")
        layer_name = p.get("benchmark_params", {}).get("layer_name", "cells")
        if not ann_path:
            logger.warning("benchmark_params.benchmark_json_path not set; skipping PR curve.")
            return

        # Compute region dims from the benchmark config (same logic as quantify.get_volume_dimensions)
        dim_dict = {
            cond: [
                cd["region"][3] - cd["region"][0],
                cd["region"][4] - cd["region"][1],
                cd["region"][5] - cd["region"][2],
            ]
            for cond, cd in benchmark_data["data"].items()
        }

        # Load ground truth annotations keyed by condition
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

            # For each detection, y_true=1 if within radius 7 of any annotation
            tree = KDTree(ann_locs, leaf_size=2)
            matches = tree.query_radius(det_locs, r=7)
            y_true = np.array([1 if len(m) > 0 else 0 for m in matches])
            pr_data[cond] = (y_true, y_score)

        if pr_data:
            plots.precision_recall_curve_plot(pr_data, p["save_path"])

    def quantify(self):
        """Compute precision/recall metrics against benchmark annotations."""
        quantify.run(self.params["benchmark_params"])

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        """Run all enabled pipeline stages in order."""
        p = self.params
        if p.get("build_model"):
            self.build_model()
        if p.get("build_training_set"):
            self.build_training_set()
        if p.get("train_model"):
            self.train_model()
        if p.get("benchmark"):
            self.benchmark()
        if p.get("quantify"):
            self.quantify()

    # ------------------------------------------------------------------
    # Default configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "SmartSPIMPipeline":
        """Instantiate the pipeline from a YAML config file.

        Parameters
        ----------
        config_path : str or Path
            Path to a YAML file whose keys match the schema in
            ``default_params()``.

        Returns
        -------
        SmartSPIMPipeline
        """
        with open(config_path) as f:
            params = yaml.safe_load(f)
        return cls(params)

    @staticmethod
    def default_params() -> dict:
        """Return the default parameter schema."""
        return {
            "build_model": True,
            "build_training_set": False,
            "train_model": True,
            "benchmark": False,
            "quantify": False,
            "Test": False,
            "save_path": "../results",
            "trained_model_path": "../data/models/model_new.keras",
            "benchmark_config_path": None,
            "yml_path": "../data",
            "annotation_params": {
                "level": 1,
                "annotation_path": "/data/annotations",
                "cells_layer": "FNs",
                "noncells_layer": "FPs",
                "resolution": [2.0, 1.8, 1.8],
                "crop_size": [52, 25, 25],
            },
            "training_sets": [
                "2026_01_09_outsource_training_set",
            ],
            "model_params": {
                "shape": (14, 14, 26, 2),
                "network_depth": "18-layer",
                "learning_rate": 3e-4,
                "optimizer": "adam",
                "number_classes": 2,
                "axis": -1,
                "starting_features": 64,
            },
            "augment_params": {
                "featurewise_center": False,
                "samplewise_center": False,
                "featurewise_std_normalization": False,
                "samplewise_std_normalization": False,
                "percentile_normalization": True,
                "percentile_range": (1, 99),
                "zca_whitening": False,
                "zca_epsilon": 1e-7,
                "rotation_range": 15,
                "width_shift_range": 0.05,
                "height_shift_range": 0.05,
                "brightness_range": (0.6, 1.5),
                "background_range": None,
                "elastic_deform": (5, 2),
                "shear_range": 5,
                "zoom_range": 0.15,
                "channel_shift_range": 0,
                "fill_mode": "reflect",
                "cval": 0,
                "horizontal_flip": True,
                "vertical_flip": True,
                "rescale": None,
                "preprocessing_function": None,
                "expand_dims": True,
                "data_format": "channels_last",
                "random_mult_range": (0.7, 1.2),
                "striping": False,
            },
            "schedule": "Cosine",
            "monitor": "val_recall_at_p90",
            "balance": False,
            "benchmark_params": {
                "classified_xml_path": "../results/classifications/",
                "benchmark_json_path": "../data/benchmark_jsons/",
                "layer_name": "cells",
            },
        }
