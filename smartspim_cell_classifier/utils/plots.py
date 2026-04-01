import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)


def precision_recall_curve_plot(pr_data: dict, save_path: str) -> None:
    """
    Plot a Precision-Recall curve for each benchmarked condition on a single figure.

    Parameters
    ----------
    pr_data : dict
        Mapping of condition name -> (y_true, y_score), where y_true is a
        binary array (1 = cell, 0 = non-cell) and y_score is the raw predicted
        probability of being a cell.
    save_path : str
        Directory where the figure will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond, (y_true, y_score) in pr_data.items():
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if y_true.sum() == 0:
            logger.warning(f"No positive labels for condition {cond}, skipping PR curve.")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{cond} (AP={ap:.2f})")
        logger.info(f"PR curve for {cond}: AP={ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()

    out_path = os.path.join(save_path, "precision_recall_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PR curve figure to {out_path}")
