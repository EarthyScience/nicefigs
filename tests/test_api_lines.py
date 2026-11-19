"""`lines`: per-series x via a mapping."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf


def test_lines_per_series_x_mapping():
    ax = nf.lines({"a": [1, 2, 3], "b": [10, 20]}, {"a": [0, 1, 2], "b": [5, 6]})
    try:
        xs = {ln.get_label(): np.asarray(ln.get_xdata()) for ln in ax.lines}
        assert xs["a"].tolist() == [1, 2, 3] and xs["b"].tolist() == [10, 20]
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)


def test_a_band_shades_between_its_bounds_along_x():
    _, ax = nf.subplots()
    x = np.array([0.0, 1.0, 2.0])
    band = nf.band(ax, x, [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], label="spread")
    ys = band.get_paths()[0].vertices[:, 1]
    assert ys.min() == 1.0 and ys.max() == 5.0
    assert band.get_label() == "spread"


def test_a_band_along_y_shades_a_profile_range():
    """``along="y"``: the band runs up the vertical coordinate and spans the horizontal one."""
    _, ax = nf.subplots()
    levels = np.array([1000.0, 500.0, 100.0])
    band = nf.band(ax, levels, [0.0, -1.0, -2.0], [2.0, 3.0, 4.0], along="y")
    xs, ys = band.get_paths()[0].vertices.T
    assert (xs.min(), xs.max()) == (-2.0, 4.0)
    assert (ys.min(), ys.max()) == (100.0, 1000.0)


def test_a_band_has_no_edge_and_draws_under_lines():
    _, ax = nf.subplots()
    band = nf.band(ax, [0, 1], [0, 0], [1, 1])
    (line,) = ax.plot([0, 1], [0.5, 0.5])
    assert band.get_linewidth()[0] == 0.0 and band.get_zorder() < line.get_zorder()
