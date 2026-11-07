"""Scorecard with the WeatherBench2-inspired diverging colormap, grouped columns, and a colour bar."""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    models = ["Persistence", "Climatology", "U-Net", "Transformer"]
    variables = ["Z500", "T850", "Q700", "T2m", "10U", "MSL"]
    groups = ["Upper-air"] * 3 + ["Surface"] * 3

    # Percent RMSE difference of each model vs. the Persistence baseline, per
    # variable — the shape a WeatherBench2-style skill scorecard compares.
    pct_diff = np.zeros((len(models), len(variables)))
    for j in range(len(variables)):
        ms = data.model_scores(seed=100 + j, models=models, metrics=["RMSE"], timescales=["daily"])
        rmse = np.array([np.mean(ms["scores"]["RMSE"][m]["daily"]) for m in models])
        pct_diff[:, j] = 100.0 * (rmse - rmse[0]) / rmse[0]

    cmap, norm = nf.scorecard_diverging_cmap()
    ax = nf.scorecard(
        pct_diff,
        row_labels=models,
        col_labels=variables,
        cmap=cmap,
        norm=norm,
        direction=dict.fromkeys(variables, "lower_better"),
        col_groups=groups,
        square=True,
        cell_format="{:+.0f}",
        colorbar=True,
        colorbar_label="RMSE difference vs. Persistence [%]",
        title="Scorecard vs. Persistence baseline",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "30_scorecard_diverging", formats=("png", "pdf"))
