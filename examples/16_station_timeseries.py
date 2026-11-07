"""Faceted station timeseries — model curtains + sparse observations per station.

Uses ``nicefigs.facet`` for the small-multiples grid and ``timeseries``'s
observation overlay + per-panel annotation.
"""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    stations = ["SPO", "MLO", "BRW", "ALT", "ASC", "CGO"]

    def draw(ax, station, i):
        dates, series = data.timeseries(n_days=365, seed=i + 1, series=["posterior", "prior"])
        rng = np.random.default_rng(i)
        idx = np.sort(rng.choice(365, 36, replace=False))
        obs = ([dates[k] for k in idx], series["posterior"][idx] + rng.normal(0, 1, 36))
        nf.timeseries(
            dates,
            series,
            ax=ax,
            obs=obs,
            obs_label="obs",
            styles={"prior": "--", "posterior": "-"},
            highlight="posterior",
            colors={"posterior": "#4477AA", "prior": "#EE6677"},
            annotation=f"bias={rng.normal(0, 0.4):+.2f}\nσ={0.5 + 0.1 * i:.2f}",
            ylabel="CO₂" if i % 3 == 0 else "",
            yunits="ppm",
            legend=False,
        )

    fig, axes = nf.facet(stations, draw, ncols=3, panel_aspect=0.62, titles=stations)
    # one shared legend for the whole figure
    import matplotlib.lines as ml

    handles = [
        ml.Line2D([], [], color="#4477AA", label="posterior"),
        ml.Line2D([], [], color="#EE6677", ls="--", label="prior"),
        ml.Line2D(
            [], [], color="#222222", marker="x", ls="None", markersize=4, markeredgewidth=0.8, label="obs"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=nf.get_theme().fs_legend,
        frameon=False,
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "16_station_timeseries", formats=("png", "pdf"))
