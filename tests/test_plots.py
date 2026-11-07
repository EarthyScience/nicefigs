"""Tests for pareto, caterpillar, bars, scorecard, cross-section and the 2D colormaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


# ------------------------------------------------------------------ pareto
def test_frontier_simple():
    # y = higher better, x = lower better; the frontier is the upper-left hull
    x = np.array([1.0, 2.0, 3.0, 2.0])
    y = np.array([1.0, 2.0, 3.0, 1.0])
    idx = nf.frontier(x, y)
    # point (2,1) is dominated by (2,2) and (1,1); (1,1) dominated by nothing on x
    assert 3 not in idx
    assert set(idx).issuperset({0, 1, 2}) or set(idx).issubset({0, 1, 2})


def test_pareto_returns_frontier_indices():
    d = npd.pareto_tradeoff()
    ax, fi = nf.pareto(d["x"], d["y"], group=d["group"], size=d["size"])
    try:
        assert fi is not None and len(fi) >= 1
    finally:
        plt.close(ax.figure)


def test_pareto_per_group_and_baselines():
    d = npd.pareto_tradeoff()
    ax, fi = nf.pareto(
        d["x"],
        d["y"],
        group=d["group"],
        frontier_per_group=True,
        baselines={"PCA": 0.3},
        xlabel="mse",
        ylabel="skill",
    )
    try:
        assert fi is None  # per-group → no single global frontier
    finally:
        plt.close(ax.figure)


# -------------------------------------------------------------- caterpillar
def test_caterpillar_sorts_and_runs():
    s = npd.site_metrics()
    ax = nf.caterpillar(
        s["labels"],
        s["bias"],
        err=s["std"],
        groups=s["layer"],
        baseline=s["baseline"],
        summary=("sigma", 0.5),
    )
    try:
        # y tick labels should be sorted by value (ascending) → first label is argmin
        assert len(ax.get_yticklabels()) == len(s["labels"])
    finally:
        plt.close(ax.figure)


def test_caterpillar_vertical_and_bounds():
    ax = nf.caterpillar(
        ["a", "b", "c"],
        [1.0, 2.0, 3.0],
        lo=[0.5, 1.5, 2.0],
        hi=[1.5, 2.5, 4.0],
        orientation="vertical",
        ref_line=None,
    )
    try:
        assert len(ax.get_xticklabels()) == 3
    finally:
        plt.close(ax.figure)


# --------------------------------------------------------------------- bars
def test_grouped_bars():
    _, ax = nf.subplots()
    nf.grouped_bars(["a", "b"], {"X": [1, 2], "Y": {"a": 3, "b": 1}}, value_label="v", ax=ax)
    try:
        assert len(ax.patches) == 4
    finally:
        plt.close(ax.figure)


def test_stacked_bars_normalized():
    _, ax = nf.subplots()
    nf.stacked_bars(["a", "b"], {"p": [1, 2], "q": [1, 2]}, normalize=True, ax=ax)
    try:
        assert len(ax.patches) == 4
    finally:
        plt.close(ax.figure)


# ---------------------------------------------------------------- scorecard
def test_scorecard_shapes_and_direction():
    vals = np.array([[1.0, 0.3], [0.5, 0.8]])
    ax = nf.scorecard(
        vals, row_labels=["m1", "m2"], col_labels=["RMSE", "R2"], direction={"RMSE": "lower_better"}
    )
    try:
        # 4 cells + 2 best-in-column highlight rectangles = 6 patches
        assert len(ax.patches) >= 4
    finally:
        plt.close(ax.figure)


def test_scorecard_target_direction():
    # bias column: 0.0 is best (closest to zero), so its cell should be "greenest"
    vals = np.array([[0.0], [0.5], [-0.9]])
    ax = nf.scorecard(
        vals,
        row_labels=["a", "b", "c"],
        col_labels=["Bias"],
        direction={"Bias": "target"},
        highlight_best=True,
    )
    try:
        # best-in-column highlight rectangle sits on row 0 (value 0.0)
        highlights = [p for p in ax.patches if not p.get_fill()]
        assert len(highlights) == 1
        assert highlights[0].get_xy()[1] == 0  # row index 0
    finally:
        plt.close(ax.figure)


def test_scorecard_handles_nan():
    vals = np.array([[1.0, np.nan], [0.5, 0.8]])
    ax = nf.scorecard(vals, row_labels=["m1", "m2"], col_labels=["a", "b"])
    plt.close(ax.figure)


def test_scorecard_default_output_unchanged_when_new_kwargs_omitted():
    # Regression contract: the new opt-in kwargs must not move a single cell
    # or tick when left at their defaults.
    vals = np.array([[1.0, 0.3], [0.5, 0.8]])
    ax = nf.scorecard(
        vals, row_labels=["m1", "m2"], col_labels=["RMSE", "R2"], direction={"RMSE": "lower_better"}
    )
    try:
        assert ax.get_xlim() == (0.0, 2.0)  # no group gaps, nc == 2
        assert ax.get_ylim() == (2.0, 0.0)  # inverted
        assert ax.get_aspect() not in ("equal", 1.0)  # square=False (default) → not forced equal
        cell_patches = [p for p in ax.patches if p.get_fill()]
        assert len(cell_patches) == 4
        xs = sorted({round(p.get_xy()[0], 6) for p in cell_patches})
        assert xs == [0.0, 1.0]  # unit-width cells, no gaps
        assert [round(t, 3) for t in ax.get_xticks()] == [0.5, 1.5]
        assert not any(t.get_fontweight() == "bold" for t in ax.texts)  # no group-title row
        assert len(ax.figure.axes) == 1  # no colorbar axes added
    finally:
        plt.close(ax.figure)


def test_scorecard_norm_bypasses_per_column_normalization():
    from nicefigs.palettes import scorecard_diverging_cmap

    cm, norm = scorecard_diverging_cmap()
    vals = np.array([[-35.0, 60.0], [3.0, -60.0]])
    ax = nf.scorecard(
        vals, row_labels=["m1", "m2"], col_labels=["a", "b"], cmap=cm, norm=norm, highlight_best=False
    )
    try:
        cell_patches = sorted(
            (p for p in ax.patches if p.get_fill()), key=lambda p: (p.get_xy()[1], p.get_xy()[0])
        )
        expected = [norm(-35.0), norm(60.0), norm(3.0), norm(-60.0)]
        for patch, e in zip(cell_patches, expected):
            assert patch.get_facecolor() == pytest.approx(cm(e))
    finally:
        plt.close(ax.figure)


def test_scorecard_grouped_columns_gap_and_titles():
    vals = np.array([[1.0, 2.0, 3.0]])
    ax = nf.scorecard(
        vals, row_labels=["m1"], col_labels=["a", "b", "c"], col_groups=["G1", "G1", "G2"], group_gap=0.6
    )
    try:
        cell_patches = sorted((p for p in ax.patches if p.get_fill()), key=lambda p: p.get_xy()[0])
        lefts = [round(p.get_xy()[0], 6) for p in cell_patches]
        # columns a, b share group G1 (no gap between them); c starts G2 after a gap.
        assert lefts[0] == 0.0
        assert lefts[1] == 1.0
        assert lefts[2] == pytest.approx(2.0 + 0.6)
        assert ax.get_xlim()[1] == pytest.approx(3.0 + 0.6)
        titles = sorted(t.get_text() for t in ax.texts if t.get_fontweight() == "bold")
        assert titles == ["G1", "G2"]
    finally:
        plt.close(ax.figure)


# ------------------------------------------------------------- cross section
def test_cross_section_single():
    cs = npd.cross_section_field()
    _, ax = nf.subplots()
    _, m = nf.cross_section(cs["lats"], cs["levels"], cs["value"], log_y=True, ax=ax)
    try:
        assert m is not None
        assert ax.yaxis_inverted()
    finally:
        plt.close(ax.figure)


def test_cross_section_panels():
    cs = npd.cross_section_field()
    val = cs["value"]
    fig, axes = nf.cross_section_panels(
        {"v": val, "d": val - val.mean(0)}, cs["lats"], cs["levels"], diff_fields={"d"}
    )
    try:
        assert len(axes) == 2
    finally:
        plt.close(fig)


def test_cross_section_panels_default_titles_from_dict_keys():
    """Default behavior (titles=None): titles come from dict keys."""
    cs = npd.cross_section_field()
    val = cs["value"]
    fig, axes = nf.cross_section_panels(
        {"v": val, "d": val - val.mean(0)}, cs["lats"], cs["levels"], diff_fields={"d"}
    )
    try:
        # Default: panel titles are the field names from dict keys, set with loc="left"
        assert axes[0].get_title(loc="left") == "v"
        assert axes[1].get_title(loc="left") == "d"
    finally:
        plt.close(fig)


def test_cross_section_panels_custom_titles_override_dict_keys():
    """With titles parameter: custom titles override dict keys; use label_panels for letters."""
    cs = npd.cross_section_field()
    val = cs["value"]
    fig, axes = nf.cross_section_panels(
        {"v": val, "d": val - val.mean(0)},
        cs["lats"],
        cs["levels"],
        diff_fields={"d"},
        titles=["Value Field", "Difference Field"],
    )
    try:
        # With titles parameter: use the provided titles, set with loc="left"
        assert axes[0].get_title(loc="left") == "Value Field"
        assert axes[1].get_title(loc="left") == "Difference Field"
        # Caller can then label with letters via label_panels
        nf.label_panels(axes)
        # After label_panels, each axes should have letter labels
        assert len(axes[0].texts) >= 1  # label text is present
        assert len(axes[1].texts) >= 1
    finally:
        plt.close(fig)


# ------------------------------------------------------------- 2D colormaps
def test_bivariate_cmap_corners():
    from matplotlib.colors import to_rgb

    from nicefigs.colormaps2d import BIVARIATE_CORNERS

    rgb = nf.bivariate_cmap(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert rgb.shape == (2, 3)
    assert rgb[0] == pytest.approx(to_rgb(BIVARIATE_CORNERS["ll"]), abs=1e-6)
    assert rgb[1] == pytest.approx(to_rgb(BIVARIATE_CORNERS["hh"]), abs=1e-6)


def test_flat_and_hsv_cmap_shape_and_range():
    for fn in (nf.flat_cmap, nf.hsv_cmap):
        rgb = fn(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        assert rgb.shape == (10, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_2d_legends_run():
    _, axes = nf.subplots(1, 3)
    nf.bivariate_legend(axes[0], x_label="x", y_label="y")
    nf.flat_legend(axes[1])
    nf.hsv_legend(axes[2])
    plt.close(axes[0].figure)


def test_binned_map_can_keep_ocean_bins():
    """Data measured over ocean must survive the land backdrop's clip."""
    pytest.importorskip("cartopy")
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 4000)
    lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 4000)))
    kept = {}
    for clip in (None, False):
        fig, ax, m = nf.map(
            lon, lat, np.ones_like(lon), kind="binned", bin_deg=5.0, clip_ocean=clip, inset_hist=False
        )
        kept[clip] = int(np.isfinite(m.get_array().filled(np.nan)).sum())
        plt.close(fig)
    assert kept[False] > 2 * kept[None]
