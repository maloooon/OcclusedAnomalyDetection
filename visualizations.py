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


# ---------------------------------------------------------------------------
# Example / demo with your numbers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
