"""Tests for modularity extensions to existing plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf
from nicefigs import data as npd


def test_metric_label_and_direction():
    assert nf.metric_label("rmse", "mm", direction=True) == "RMSE [mm] ↓"
    assert nf.metric_label("r2", direction=True) == "R² ↑"
    assert nf.metric_label("unknown_metric") == "Unknown Metric"
    assert nf.axis_label("x", "m2", direction="higher_better") == "x [m²] ↑"


def test_reference_line_hv():
    _, ax = nf.subplots()
    nf.reference_line(ax, 0.5, axis="y", label="chance", fmt="{label} ({value:.2f})")
    nf.reference_line(ax, 1.0, axis="x")
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_facet_hides_unused_and_labels():
    calls = []
    fig, axes = nf.facet(range(5), lambda ax, item, i: calls.append(i), ncols=3, labels=True)
    try:
        assert calls == [0, 1, 2, 3, 4]
        assert len(axes) == 6  # 2x3 grid
        assert not axes[5].axison  # last cell hidden
    finally:
        plt.close(fig)


def test_grid_with_legend_row():
    fig, axes, ax_legend = nf.grid(1, 2, with_legend_row=True)
    try:
        assert axes.shape == (1, 2)
        assert not ax_legend.axison
    finally:
        plt.close(fig)


def test_timeseries_obs_and_annotation():
    dates, series = npd.timeseries(n_days=120)
    obs = (dates[::5], series["observed"][::5])
    ax = nf.timeseries(dates, series, obs=obs, annotation="bias=0.1", styles={"model A": "--"})
    try:
        # obs scatter present
        assert len(ax.collections) >= 1
    finally:
        plt.close(ax.figure)


def test_scatter_errorbars_and_labels():
    _, ax = nf.subplots()
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
    nf.scatter(x, y, xerr=0.1, yerr=0.2, point_labels=[f"s{i}" for i in range(20)], label_top=3, ax=ax)
    try:
        texts = [t for t in ax.texts]
        assert len(texts) == 3
    finally:
        plt.close(ax.figure)


def test_image_overlay_contour():
    _, ax = nf.subplots()
    img = npd.regional_grid()[2]
    mask = (img > img.mean()).astype(float)
    nf.image(img, overlay_contour=mask, robust=True, ax=ax)
    plt.close(ax.figure)


def test_pca_rgb_shape_and_shared_basis():
    field = np.random.default_rng(0).normal(0, 1, (12, 10, 6))
    rgb, basis = nf.pca_rgb(field, return_basis=True)
    assert rgb.shape == (12, 10, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1
    rgb2 = nf.pca_rgb(field, basis=basis)
    assert rgb2.shape == (12, 10, 3)


def test_pca_rgb_shared_basis_is_comparable_across_fields():
    """A shared basis fixes the colour scale, so two fields of the same scene
    stay comparable — the bright half must not be renormalised to the full range."""
    rng = np.random.default_rng(1)
    scene = rng.normal(0, 1, (40, 40, 5))
    _, basis = nf.pca_rgb(scene, return_basis=True)
    bright = nf.pca_rgb(scene[:20], basis=basis)
    dim = nf.pca_rgb(scene[:20] * 0.1, basis=basis)
    assert bright.mean() != dim.mean()
    # A two-element basis (mean, components) keeps the old per-field behaviour.
    legacy = nf.pca_rgb(scene[:20], basis=basis[:2])
    assert legacy.min() == 0.0 and legacy.max() == 1.0


def test_map_fill_bins_runs():
    lon, lat, v = npd.global_points(n=6000)
    fig, ax, m = nf.map(lon, lat, v, kind="binned", fill_bins=1, bin_deg=4.0, inset_hist=False)
    assert m is not None
    plt.close(fig)
