"""Ternary scatter — soil texture compositions coloured by their mixture."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    comp, _ = data.ternary_points(n=900)
    ax, _ = nf.ternary_scatter(comp, labels=("Sand", "Silt", "Clay"), size=12)
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "11_ternary", formats=("png", "pdf"))
