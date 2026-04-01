import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def precision_recall_curve_plot(pr_data: dict, plots_path: str) -> None:
    """
    Plot a Precision-Recall curve for each benchmarked condition on a single figure.

    Parameters
    ----------
    pr_data : dict
        Mapping of condition name -> (y_true, y_score), where y_true is a
        binary array (1 = cell, 0 = non-cell) and y_score is the raw predicted
        probability of being a cell.
    plots_path : str
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

    out_path = os.path.join(plots_path, "precision_recall_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PR curve figure to {out_path}")


def metrics_barplot(metric_df, plots_path: str) -> None:
    """
    Bar plot of Precision, Recall, and F1 for each condition plus the average.

    Parameters
    ----------
    metric_df : pd.DataFrame
        DataFrame with columns 'Condition', 'Precision', 'Recall', 'F1'.
    plots_path : str
        Directory where the figure will be saved.
    """
    conditions = list(metric_df["Condition"])
    precision = list(metric_df["Precision"])
    recall = list(metric_df["Recall"])
    f1 = list(metric_df["F1"])

    conditions.append("Average")
    precision.append(np.mean(metric_df["Precision"]))
    recall.append(np.mean(metric_df["Recall"]))
    f1.append(np.mean(metric_df["F1"]))

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.4), 6))

    bars_p = ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    bars_r = ax.bar(x,          recall,    width, label="Recall",    color="#DD8452")
    bars_f = ax.bar(x + width,  f1,        width, label="F1",        color="#55A868")

    ax.axvline(x[-1] - width * 2, color="grey", linestyle="--", linewidth=0.8)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1 by Condition")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(plots_path, "metrics_barplot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved metrics bar plot to {out_path}")


def tp_fp_fn_barplot(metric_df, plots_path: str) -> None:
    """
    Stacked bar chart of TP, FP, and FN counts for each condition.

    Parameters
    ----------
    metric_df : pd.DataFrame
        DataFrame with columns 'Condition', 'TP', 'FP', 'FN'.
    plots_path : str
        Directory where the figure will be saved.
    """
    conditions = list(metric_df["Condition"])
    tp = list(metric_df["TP"])
    fp = list(metric_df["FP"])
    fn = list(metric_df["FN"])

    x = np.arange(len(conditions))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.4), 6))

    bars_tp = ax.bar(x, tp, width, label="TP", color="#55A868")
    bars_fp = ax.bar(x, fp, width, bottom=tp, label="FP", color="#DD8452")
    bars_fn = ax.bar(x, fn, width, bottom=[t + f for t, f in zip(tp, fp)], label="FN", color="#C44E52")

    for bars in (bars_tp, bars_fp, bars_fn):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cell Count")
    ax.set_title("TP / FP / FN Counts by Condition")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(plots_path, "tp_fp_fn_barplot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved TP/FP/FN bar plot to {out_path}")


def cell_count_comparison_plot(metric_df, plots_path: str) -> None:
    """
    Grouped bar chart comparing annotated vs detected cell counts per condition.

    Parameters
    ----------
    metric_df : pd.DataFrame
        DataFrame with columns 'Condition', 'Total_Annotations', 'Total_Classifications'.
    plots_path : str
        Directory where the figure will be saved.
    """
    conditions = list(metric_df["Condition"])
    annotated = list(metric_df["Total_Annotations"])
    detected = list(metric_df["Total_Classifications"])

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.4), 6))

    bars_a = ax.bar(x - width / 2, annotated, width, label="Annotated", color="#4C72B0")
    bars_d = ax.bar(x + width / 2, detected,  width, label="Detected",  color="#DD8452")

    for bars in (bars_a, bars_d):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cell Count")
    ax.set_title("Annotated vs Detected Cell Counts by Condition")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(plots_path, "cell_count_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved cell count comparison plot to {out_path}")


def score_distribution_plot(pr_data: dict, plots_path: str) -> None:
    """
    Histogram of predicted probabilities split by TP vs FP for each condition.

    Parameters
    ----------
    pr_data : dict
        Mapping of condition name -> (y_true, y_score).
    plots_path : str
        Directory where the figure will be saved.
    """
    n = len(pr_data)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, (cond, (y_true, y_score)) in zip(axes.flat, pr_data.items()):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        tp_scores = y_score[y_true == 1]
        fp_scores = y_score[y_true == 0]

        bins = np.linspace(0, 1, 30)
        ax.hist(tp_scores, bins=bins, alpha=0.7, label=f"TP (n={len(tp_scores)})", color="#55A868")
        ax.hist(fp_scores, bins=bins, alpha=0.7, label=f"FP (n={len(fp_scores)})", color="#DD8452")
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    # Hide any unused subplots
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Score Distributions (TP vs FP) by Condition", fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(plots_path, "score_distributions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved score distribution plot to {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_plots(plots_path: str, metric_df=None, pr_data=None) -> None:
    """
    Run all benchmark plots and save them to plots_path.

    Each plot is skipped gracefully if the required data is not provided.

    Parameters
    ----------
    plots_path : str
        Directory where all figures are saved (created if needed).
    metric_df : pd.DataFrame, optional
        Output of quantify.run() — drives metrics_barplot, tp_fp_fn_barplot,
        cell_count_comparison_plot.
    pr_data : dict, optional
        Mapping cond -> (y_true, y_score) — drives precision_recall_curve_plot
        and score_distribution_plot.
    """
    os.makedirs(plots_path, exist_ok=True)

    if pr_data:
        precision_recall_curve_plot(pr_data, plots_path)
        score_distribution_plot(pr_data, plots_path)

    if metric_df is not None and not metric_df.empty:
        metrics_barplot(metric_df, plots_path)
        tp_fp_fn_barplot(metric_df, plots_path)
        cell_count_comparison_plot(metric_df, plots_path)
