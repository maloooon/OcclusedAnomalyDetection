"""
Plot the evolution of segmentation metrics as successive post-processing
filters are stacked on top of the SAM-b baseline.

Designed for the RaspGrade segmentation benchmark: each "stage" corresponds
to the cumulative addition of a filter (red filter, bbox filter, hole filter,
overlap filter, ...). The function returns a matplotlib Figure so it can
be saved as PDF for the thesis or shown inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FilterStage:
    """
    One row of the evolution table: a named stage with its metrics.

    All "ratio" metrics live in [0, 1]. Inference time is in seconds,
    model size in millions of parameters. Both ratio metrics and the
    cost metrics may be None for stages where the value is not yet
    available (e.g. inference time only measured at the baseline).
    """
    name: str
    iou: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    ap50: Optional[float] = None
    ap50_95: Optional[float] = None
    inference_time_s: Optional[float] = None
    model_size_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_filter_evolution(
    stages: list[FilterStage],
    *,
    title: str = "Effect of cumulative post-processing filters, YOLO-26n-seg fine-tuned",
    figsize: tuple[float, float] = (10, 7),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how segmentation metrics evolve across a sequence of filter stages.

    Default layout is two stacked subplots sharing the x-axis:
      - Top panel: ratio metrics (IoU, Precision, Recall, F1, AP@50, AP@50:95)
        plotted on a single [0, 1] axis, since they share units.
      - Bottom panel: inference time (left y-axis) and model size
        (right twin y-axis), since these live on different scales from
        each other and from the ratio metrics.


    Parameters
    ----------
    stages : list[FilterStage]
        Ordered list of stages. The order on the x-axis follows the
        order of this list (so put the baseline first and the most
        aggressive filter combination last).
    title : str
        Figure suptitle.
    figsize : (float, float)
        Figure size in inches.
    savepath : str | None
        If given, the figure is saved to this path (e.g. "evolution.pdf").

    Returns
    -------
    matplotlib.figure.Figure
        The created figure. Useful if the caller wants to tweak it
        further before saving.
    """

    # --- 1. Pull the x-axis labels and a helper for missing values ----------
    # Each stage label goes on the x-axis. Stages without a metric value
    # contribute None at that position, which matplotlib draws as a gap
    # only if we feed it NaN — so we convert None -> np.nan up front.
    x_labels = [s.name for s in stages]
    x_pos = np.arange(len(stages))

    def col(attr: str) -> np.ndarray:
        """Extract one metric across all stages as a numpy array with NaN for missing."""
        return np.array(
            [getattr(s, attr) if getattr(s, attr) is not None else np.nan for s in stages],
            dtype=float,
        )

    # All ratio-valued metrics share a common scale [0, 1].
    ratio_metrics = {
        "IoU":       col("iou"),
        "Precision": col("precision"),
        "Recall":    col("recall"),
        "F1":        col("f1"),
        "AP@50":     col("ap50"),
        "AP@50:95":  col("ap50_95"),
    }
    inf_time = col("inference_time_s")
    size_m   = col("model_size_m")

    # Distinct colors and markers so the lines stay legible in B&W print too.
    # Using tab10 is fine for six lines; markers add a second visual channel.
    style = {
        "IoU":       ("#1f77b4", "o"),
        "Precision": ("#ff7f0e", "s"),
        "Recall":    ("#2ca02c", "^"),
        "F1":        ("#d62728", "D"),
        "AP@50":     ("#9467bd", "v"),
        "AP@50:95":  ("#8c564b", "P"),
    }

    # gridspec_kw shrinks the cost panel — the ratio panel carries the
    # main story so it gets more vertical space.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Top: ratio metrics ---------------------------------------------
    for name, values in ratio_metrics.items():
        color, marker = style[name]
        ax_top.plot(
            x_pos, values,
            label=name, color=color, marker=marker,
            linewidth=2, markersize=7,
        )
    ax_top.set_ylim(0.75, 1.0)
    ax_top.set_ylabel("Metric value")
    ax_top.grid(True, alpha=0.3)
    # Legend outside the plot keeps the lines readable when many stages
    # produce values close together (e.g. IoU vs Recall around 0.82).
    ax_top.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    # --- Bottom: cost metrics on twin axes -------------------------------
    # Left axis = inference time (s), right axis = model size (M params).
    # Bars (rather than lines) make the per-stage value easy to read and
    # also make missing values visually obvious — the bar is simply absent.
    bar_width = 0.35
    ax_bot.bar(
        x_pos - bar_width / 2, inf_time,
        width=bar_width, color="#1f77b4", alpha=0.7,
        label="Inference time (s)",
    )
    ax_bot.set_ylabel("Inference time (s)", color="#1f77b4")
    ax_bot.tick_params(axis="y", labelcolor="#1f77b4")
    ax_bot.grid(True, axis="y", alpha=0.3)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.bar(
        x_pos + bar_width / 2, size_m,
        width=bar_width, color="#d62728", alpha=0.7,
        label="Model size (M params)",
    )
    ax_bot_r.set_ylabel("Model size (M params)", color="#d62728")
    ax_bot_r.tick_params(axis="y", labelcolor="#d62728")

    # Combine the two bar legends into one block under the ratio legend.
    bars_left,  labs_left  = ax_bot.get_legend_handles_labels()
    bars_right, labs_right = ax_bot_r.get_legend_handles_labels()
    ax_bot_r.legend(
        bars_left + bars_right, labs_left + labs_right,
        loc="center left",
        bbox_to_anchor=(1.10, 0.5),  # nudge clear of the right y-axis
        frameon=False,
    )

    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels(x_labels, rotation=20, ha="right")
    ax_bot.set_xlabel("Cumulative filter stage")



    if savepath is not None:
        # bbox_inches="tight" stops the external legend from getting clipped.
        fig.savefig(savepath, bbox_inches="tight", dpi=200)

    return fig




def plot_auroc_auprc_vs_k(
    auroc_scores: list[float],
    auprc_scores: list[float],
    base_auroc: float,
    base_auprc: float,
    *,
    k_values: list[float] = [0.01, 0.04, 0.07, 0.10],
    title: str = "AUROC / AUPRC on increasing k (topk_mean) values",
    figsize: tuple[float, float] = (7, 5),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Plot AUROC and AUPRC as a function of occlusion ratio k, with baselines.

    Parameters
    ----------
    auroc_scores : list[float]
        AUROC values (in [0, 1]) at each k in *k_values*, same order.
    auprc_scores : list[float]
        AUPRC values (in [0, 1]) at each k in *k_values*, same order.
    base_auroc : float
        Baseline AUROC (drawn as a horizontal line across the full x range).
    base_auprc : float
        Baseline AUPRC (drawn as a horizontal line across the full x range).
    k_values : list[float]
        Occlusion ratios on the x-axis; defaults to [0.01, 0.04, 0.07, 0.10].
    title : str
        Figure title.
    figsize : (float, float)
        Figure size in inches.
    savepath : str | None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(auroc_scores) != len(k_values) or len(auprc_scores) != len(k_values):
        raise ValueError(
            f"auroc_scores and auprc_scores must each have {len(k_values)} entries "
            f"(one per k value)."
        )

    # x-axis in percentage points (1 %, 4 %, …)
    x_pct = [k * 100 for k in k_values]

    # Convert scores to percentage for the y-axis
    auroc_pct = [v * 100 for v in auroc_scores]
    auprc_pct = [v * 100 for v in auprc_scores]
    base_auroc_pct = base_auroc * 100
    base_auprc_pct = base_auprc * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline horizontal lines span the full x range with a small margin
    x_min, x_max = x_pct[0] - 0.5, x_pct[-1] + 0.5
    ax.hlines(
        base_auroc_pct, x_min, x_max,
        colors="#1f77b4", linestyles="--", linewidth=1.5,
        label=f"Baseline AUROC ({base_auroc_pct:.1f} %)",
    )
    ax.hlines(
        base_auprc_pct, x_min, x_max,
        colors="#ff7f0e", linestyles="--", linewidth=1.5,
        label=f"Baseline AUPRC ({base_auprc_pct:.1f} %)",
    )

    # Connected lines for each k
    ax.plot(
        x_pct, auroc_pct,
        color="#1f77b4", marker="o", linewidth=2, markersize=7,
        label="AUROC",
    )
    ax.plot(
        x_pct, auprc_pct,
        color="#ff7f0e", marker="s", linewidth=2, markersize=7,
        label="AUPRC",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_pct)
    ax.set_xticklabels([f"{x:.0f} %" for x in x_pct])
    ax.set_xlabel("k values")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)

    return fig



@dataclass
class ModelScores:
    """AUROC / AUPRC means and stds for one model, before and after a change."""
    name: str
    initial_auroc_mean: float
    initial_auroc_std: float
    final_auroc_mean: float
    final_auroc_std: float
    initial_auprc_mean: float
    initial_auprc_std: float
    final_auprc_mean: float
    final_auprc_std: float


def plot_initial_vs_final(
    model_scores: list[ModelScores],
    *,
    title: str = "Initial vs. Final AUROC and AUPRC by Model",
    figsize: tuple[float, float] = (14, 6),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side grouped bar chart comparing initial and final AUROC / AUPRC
    for each model.

    Parameters
    ----------
    model_scores : list[ModelScores]
        One entry per model. Order determines the x-axis order.
    title : str
        Figure suptitle.
    figsize : (float, float)
        Figure size in inches.
    savepath : str | None
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(model_scores)
    x = np.arange(n)
    bar_w = 0.34

    COLOR_INIT  = "#9ECAE1"   # light blue  — initial
    COLOR_FINAL = "#08519C"   # deep blue   — final
    ECOLOR      = "#333333"   # error-bar caps

    names = [s.name for s in model_scores]

    def _arrays(attr_mean, attr_std):
        means = np.array([getattr(s, attr_mean) for s in model_scores])
        stds  = np.array([getattr(s, attr_std)  for s in model_scores])
        return means, stds

    init_auroc,  init_auroc_std  = _arrays("initial_auroc_mean", "initial_auroc_std")
    final_auroc, final_auroc_std = _arrays("final_auroc_mean",   "final_auroc_std")
    init_auprc,  init_auprc_std  = _arrays("initial_auprc_mean", "initial_auprc_std")
    final_auprc, final_auprc_std = _arrays("final_auprc_mean",   "final_auprc_std")

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=figsize)

    def _draw_panel(ax, init_vals, init_stds, final_vals, final_stds, ylabel):
        err_kw = dict(ecolor=ECOLOR, capsize=4, error_kw=dict(capthick=1.2, elinewidth=1.2))

        bars_i = ax.bar(
            x - bar_w / 2, init_vals, bar_w,
            yerr=init_stds, label="Initial",
            color=COLOR_INIT,  alpha=0.92, zorder=3, **err_kw,
        )
        bars_f = ax.bar(
            x + bar_w / 2, final_vals, bar_w,
            yerr=final_stds, label="Final",
            color=COLOR_FINAL, alpha=0.92, zorder=3, **err_kw,
        )

        # Value labels on top of each bar
        for bar, val, std in zip(bars_i, init_vals, init_stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.4,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, color="#555555",
            )
        for bar, val, std in zip(bars_f, final_vals, final_stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.4,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, color=COLOR_FINAL,
                fontweight="bold",
            )

        all_vals = np.concatenate([init_vals - init_stds, final_vals - final_stds])
        all_tops = np.concatenate([init_vals + init_stds, final_vals + final_stds])
        margin = (all_tops.max() - all_vals.min()) * 0.18
        ax.set_ylim(max(0.0, all_vals.min() - margin), min(100.0, all_tops.max() + margin + 4.0))

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", pad=6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)

    _draw_panel(ax_roc, init_auroc, init_auroc_std, final_auroc, final_auroc_std, "AUROC")
    _draw_panel(ax_prc, init_auprc, init_auprc_std, final_auprc, final_auprc_std, "AUPRC")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=3.0)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)

    return fig


if __name__ == "__main__":
    '''
    stages_samb = [
        FilterStage(
            name="baseline",
            iou=0.9224, precision=0.2614, recall=0.8257, f1=0.3956,
            ap50=0.5409, ap50_95=0.4750,
            inference_time_s=1.24, model_size_m=94.0,
        ),
        FilterStage(
            name="+ point prompts",
            iou=0.9218, precision=0.3427, recall=0.8257, f1=0.4820,
            ap50=0.6245, ap50_95=0.5478,
            inference_time_s=0.83, model_size_m=96.7,
        ),
        FilterStage(
            name="+ red filter",
            iou=0.9218, precision=0.7915, recall=0.8257, f1=0.8025,
            ap50=0.7620, ap50_95=0.6699,
            inference_time_s=0.86, model_size_m=96.7,
        ),
        FilterStage(
            name="+ bbox size filter",
            iou=0.9222, precision=0.8171, recall=0.8238, f1=0.8177,
            ap50=0.7654, ap50_95=0.6729,
            inference_time_s=0.86, model_size_m=96.7,
        ),
        FilterStage(
            name="+ hole/island filter",
            iou=0.9223, precision=0.8173, recall=0.8239, f1=0.8179,
            ap50=0.7654, ap50_95=0.6731,
            inference_time_s=1.21, model_size_m=96.7,
        ),
        FilterStage(
            name="+ overlap filter",
            iou=0.9154, precision=0.9174, recall=0.8142, f1=0.8607,
            ap50=0.7681, ap50_95=0.6649,
            inference_time_s=1.39, model_size_m=194.7,
        ),
    ]


    stages_yolo = [
    FilterStage(
        name="baseline",
        iou=0.9175, precision=0.9414, recall=0.9342, f1=0.9374,
        ap50=0.9379, ap50_95=.8072,
        inference_time_s=0.02, model_size_m=2.7,
    ),
    FilterStage(
        name="+ red filter",
        iou=0.9175, precision=0.9414, recall=0.9342, f1=0.9374,
        ap50=0.9379, ap50_95=0.8072,
        inference_time_s=0.04, model_size_m=2.7,
    ),
    FilterStage(
        name="+ bbox size filter",
        iou=0.9175, precision=0.9414, recall=0.9342, f1=0.9374,
        ap50=0.9379, ap50_95=0.8072,
        inference_time_s=0.04, model_size_m=2.7,
    ),
    FilterStage(
        name="+ hole/island filter",
        iou=0.9179, precision=0.9418, recall=0.9342, f1=0.9377,
        ap50=0.9379, ap50_95=0.8077,
        inference_time_s=0.34, model_size_m=2.7,
    ),
    FilterStage(
        name="+ overlap filter",
        iou=0.9179, precision=0.9568, recall=0.9343, f1=0.9452,
        ap50=0.9384, ap50_95=0.8080,
        inference_time_s=0.46, model_size_m=100.7,
    ),
]


    plot_filter_evolution(stages_yolo, savepath="filter_evolution.png")
    '''

   # plot_auroc_auprc_vs_k(
   #     auroc_scores=[0.8297, 0.8448, 0.8504, 0.8534],
   #     auprc_scores=[0.8496, 0.8632, 0.8682, 0.8720],
   #     base_auroc=0.8277,
   #     base_auprc=0.8406,
   #     savepath="auroc_auprc_vs_k_rd4ad.png",
   # )


   # plot_auroc_auprc_vs_k(
    #    auroc_scores=[0.7302, 0.7315, 0.7329, 0.7345],
    #    auprc_scores=[0.7473, 0.7489, 0.7503, 0.7512],
    #    base_auroc=0.7166,
    #    base_auprc=0.7388,
    #    savepath="auroc_auprc_vs_k_ssn.png",
    #)

    scores = [
        ModelScores(
            name="PatchCore",
            initial_auroc_mean=77.58, initial_auroc_std=1.24,
            final_auroc_mean=89.71,   final_auroc_std=1.04,
            initial_auprc_mean=77.50, initial_auprc_std=0.83,
            final_auprc_mean=89.75,   final_auprc_std=1.50,
        ),
        ModelScores(
            name="STFPM",
            initial_auroc_mean=75.03, initial_auroc_std=0.97,
            final_auroc_mean=92.07,   final_auroc_std=1.28,
            initial_auprc_mean=76.85, initial_auprc_std=1.33,
            final_auprc_mean=92.66,   final_auprc_std=1.83,
        ),
        ModelScores(
            name="RD4AD",
            initial_auroc_mean=80.44, initial_auroc_std=2.81,
            final_auroc_mean=85.43,   final_auroc_std=2.90,
            initial_auprc_mean=80.40, initial_auprc_std=2.65,
            final_auprc_mean=87.20,   final_auprc_std=3.24,
        ),
        ModelScores(
            name="CFA",
            initial_auroc_mean=56.21, initial_auroc_std=4.16,
            final_auroc_mean=82.50,   final_auroc_std=1.84,
            initial_auprc_mean=56.64, initial_auprc_std=5.49,
            final_auprc_mean=83.08,   final_auprc_std=1.89,
        ),
        ModelScores(
            name="GANomaly",
            initial_auroc_mean=73.80, initial_auroc_std=2.50,
            final_auroc_mean=75.08,   final_auroc_std=3.69,
            initial_auprc_mean=74.81, initial_auprc_std=2.39,
            final_auprc_mean=75.40,   final_auprc_std=4.17,
        ),
        ModelScores(
            name="FastFlow",
            initial_auroc_mean=66.05, initial_auroc_std=2.71,
            final_auroc_mean=76.33,   final_auroc_std=2.28,
            initial_auprc_mean=66.99, initial_auprc_std=2.62,
            final_auprc_mean=76.95,   final_auprc_std=1.05,
        ),
        ModelScores(
            name="SuperSimpleNet",
            initial_auroc_mean=65.93, initial_auroc_std=3.30,
            final_auroc_mean=75.25,   final_auroc_std=2.11,
            initial_auprc_mean=63.73, initial_auprc_std=2.29,
            final_auprc_mean=73.81,   final_auprc_std=2.82,
        ),
        ModelScores(
            name="SINBAD",
            initial_auroc_mean=72.22, initial_auroc_std=1.61,
            final_auroc_mean=78.65,   final_auroc_std=0.61,
            initial_auprc_mean=70.34, initial_auprc_std=1.39,
            final_auprc_mean=79.49,   final_auprc_std=1.16,
        ),
    ]

    plot_initial_vs_final(scores, savepath="initial_vs_final.png")
