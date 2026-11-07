"""Caterpillar / forest plot — per-site bias ± σ, ranked, with a baseline model."""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    s = data.site_metrics(n_sites=26)
    ax = nf.caterpillar(
        s["labels"],
        s["bias"],
        err=s["std"],
        groups=s["layer"],
        baseline=s["baseline"],
        baseline_label="GEOS",
        value_label="bias",
        value_units="ppmv",
        summary=("σ̄", float(np.sqrt(np.mean(s["std"] ** 2)))),
        ref_line=0.0,
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "19_caterpillar", formats=("png", "pdf"))
