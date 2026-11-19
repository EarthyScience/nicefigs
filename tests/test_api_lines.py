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


def test_a_band_takes_a_hatch_and_an_outline_on_request():
    _, ax = nf.subplots()
    band = nf.band(ax, [0, 1], [0, 0], [1, 1], hatch="//", linewidth=0.8)
    assert band.get_hatch() == "//" and band.get_linewidth()[0] == 0.8


def test_a_band_spans_descending_coordinates_and_leaves_gaps_at_nan():
    _, ax = nf.subplots()
    levels = np.array([1000.0, 700.0, 500.0, 300.0, 100.0])
    low = np.array([0.0, np.nan, 1.0, 1.0, 1.0])
    band = nf.band(ax, levels, low, low + 1.0, along="y")
    ys = np.concatenate([p.vertices[:, 1] for p in band.get_paths()])
    assert ys.max() == 1000.0 and ys.min() == 100.0
    assert not np.isnan(np.concatenate([p.vertices for p in band.get_paths()])).any()
    assert len(band.get_paths()) == 2  # the NaN level splits the band


def _bands(ax):
    from matplotlib.collections import PolyCollection

    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_lines_timeseries_and_profile_draw_their_bands_flat_and_under_the_lines():
    x = np.arange(5.0)
    drawn = [
        nf.lines(x, {"a": x}, bands={"a": (x - 1, x + 1)}),
        nf.timeseries(
            np.arange("2022-01-01", "2022-01-06", dtype="datetime64[D]"),
            {"a": x},
            bands={"a": (x - 1, x + 1)},
        ),
        nf.profile({"a": x}, x, bands={"a": (x - 1, x + 1)}),
    ]
    for ax in drawn:
        (band,) = _bands(ax)
        assert band.get_linewidth()[0] == 0.0
        assert band.get_zorder() == 1 and all(line.get_zorder() > 1 for line in ax.lines)
        plt.close(ax.figure)
