"""Difference map with significance stippling.

A diverging regional difference map; ``hatch_region(..., style="dots")`` stipples the
cells where the difference is significant, with dots only and no outline, so the colours
underneath stay readable.
"""

import numpy as np

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    lon, lat, diff = data.regional_grid(res=0.25, extent=(-12, 40, 34, 60))
    significant = np.abs(diff) > 0.12
    fig, ax, _ = nf.map(
        lon,
        lat,
        diff,
        extent=(-12, 40, 34, 60),
        projection="platecarree",
        diverging=True,
        inset_hist=False,
        cbar=True,
        borders=True,
        colorbar_label="model difference [K]",
        title="Difference, stippled where significant",
    )
    nf.hatch_region(ax, lon.ravel(), lat.ravel(), significant.ravel(), style="dots", bin_deg=0.5)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "36_stippled_difference_map", formats=("png", "pdf"))
