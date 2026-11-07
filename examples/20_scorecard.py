"""Scorecard — a model×metric table coloured per column by how good each value is."""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    ms = data.model_scores()
    models, metrics = ms["models"], ms["metrics"]
    vals = np.array([[np.mean(ms["scores"][m][mdl]["daily"]) for m in metrics] for mdl in models])
    ax = nf.scorecard(
        vals,
        row_labels=models,
        col_labels=metrics,
        direction={"RMSE": "lower_better", "Bias": "target", "R$^2$": "higher_better"},
        col_units={"RMSE": "mm"},
        title="Head-to-head scorecard",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "20_scorecard", formats=("png", "pdf"))
