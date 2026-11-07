"""Scatter with three encodings — marker shape (group), size, and colour."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    d = data.scatter_bubbles(n=140)
    fig, ax = nf.subplots(width="half", aspect=0.82)
    nf.scatter(
        d["x"],
        d["y"],
        group=d["group"],
        size=d["size"],
        size_label="area [km²]",
        alpha=0.8,
        ax=ax,
    )
    nf.set_labels(ax, "predictor", "response")
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "13_scatter", formats=("png", "pdf"))
