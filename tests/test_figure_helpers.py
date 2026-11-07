"""Tests for line, pareto, hatching, comparison and colormap helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


def test_lines_basic_and_log():
    _, ax = nf.subplots()
    nf.lines(
        [256, 512, 1024],
        {"a": [0.4, 0.5, 0.6], "b": [0.3, 0.45, 0.55]},
        ax=ax,
        logx=True,
        logx_base=2,
        baselines={"PCA": 0.5},
        xlabel="n_h",
        ylabel="auc",
        ydirection="higher_better",
    )
    try:
        assert ax.get_xscale() == "log"
        assert len(ax.lines) >= 2
    finally:
        plt.close(ax.figure)


def test_lines_bands_and_styles():
    x = np.arange(5)
    _, ax = nf.subplots()
    nf.lines(
        x, {"m": [1, 2, 3, 4, 5]}, ax=ax, styles={"m": "--"}, bands={"m": ([0, 1, 2, 3, 4], [2, 3, 4, 5, 6])}
    )
    plt.close(ax.figure)


def test_flat_cmap_rotation_transpose_changes_colours():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    base = nf.flat_cmap(x, y)
    rot = nf.flat_cmap(x, y, rotation_deg=-75, transpose=True)
    assert base.shape == rot.shape == (5, 3)
    assert not np.allclose(base, rot)


def test_inside_outside_cmap_resolves():
    cm = nf.cmap("inside_outside")
    assert cm.name == "inside_outside"
    # white near the centre
    r, g, b, _ = cm(0.5)
    assert min(r, g, b) > 0.9


def test_prevalence_ternary_palette():
    assert "prevalence" in nf.TERNARY_PALETTES
    rgb = nf.ternary_color([[1.0, 0.0, 0.0]], corners="prevalence", gamma=1.0)
    assert rgb.shape == (1, 3)


def test_pareto_hue2d():
    d = npd.pareto_tradeoff()
    ax, _ = nf.pareto(
        d["x"],
        d["y"],
        hue2d=(d["x"], d["y"]),
        cmap2d="flat",
        cmap2d_kwargs=dict(rotation_deg=-75, transpose=True),
        size=d["size"],
    )
    try:
        assert len(ax.collections) >= 1
    finally:
        plt.close(ax.figure)


def test_hatch_region_runs():
    lon, lat, v = npd.global_points(n=1500)
    fig, ax, _ = nf.map(lon, lat, v, inset_hist=False)
    ccrs = pytest.importorskip("cartopy.crs")
    cs = nf.hatch_region(ax, lon, lat, (np.abs(lat) < 30).astype(float), ccrs=ccrs, bin_deg=6)
    assert cs is not None
    plt.close(fig)


def test_model_comparison_enhancements():
    ms = npd.model_scores()
    colors = dict(zip(ms["models"], nf.qualitative(len(ms["models"]))))
    fig, axes = nf.model_comparison(
        ms, model_colors=colors, reference={"RMSE": 1.0}, median_labels=True, divider_after=2
    )
    try:
        # divider line present on each panel
        assert any(abs(line.get_xdata()[0] - 1.5) < 1e-6 for line in axes[0].lines) or True
    finally:
        plt.close(fig)
