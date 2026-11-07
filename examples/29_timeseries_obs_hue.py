"""Timeseries — sparse observations coloured by a third variable (e.g. altitude)."""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    dates, series = data.timeseries(n_days=200, series=["model"])
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(dates), 60, replace=False))
    odates = [dates[i] for i in idx]
    ovals = series["model"][idx] + rng.normal(0, 1, 60)
    altitude = rng.uniform(0, 12000, 60)

    ax = nf.timeseries(
        dates,
        series,
        obs=(odates, ovals),
        obs_hue=altitude,
        obs_cmap="viridis",
        obs_hue_label="altitude [m]",
        obs_colorbar=True,
        ylabel="CO2",
        yunits="ppm",
        title="Aircraft profile vs a model curtain, coloured by altitude",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "29_timeseries_obs_hue", formats=("png", "pdf"))
