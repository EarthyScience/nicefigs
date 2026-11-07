"""Timeseries — observations vs models with an uncertainty band and date axis."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    dates, series = data.timeseries(n_days=730)
    # a simple uncertainty band around model A
    a = series["model A"]
    bands = {"model A": (a - 3, a + 3)}
    ax = nf.timeseries(
        dates,
        series,
        bands=bands,
        highlight="observed",
        ylabel="air temperature",
        yunits="degC",
        title="Daily air temperature",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "15_timeseries", formats=("png", "pdf"))
