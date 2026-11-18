"""polygon_mesh, outline_boxes, one_to_one, kde_contours, scatter_regression, histogram and colorbar(cax=)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf


def _hexagon(cx: float, cy: float) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, 7)[:-1]
    return np.stack([cx + np.cos(a), cy + np.sin(a)], axis=-1)


def _triangle(cx: float, cy: float) -> np.ndarray:
    return np.array([[cx, cy], [cx + 1, cy], [cx, cy + 1]])


def test_polygon_mesh_by_value_maps_through_the_limits_and_mixes_corner_counts():
    fig, ax = plt.subplots()
    cells = [_hexagon(0, 0), _triangle(3, 0)]
    coll = nf.polygon_mesh(ax, cells, [0.0, 1.0], cmap="viridis", vmin=0.0, vmax=2.0)
    assert np.allclose(coll.get_array(), [0.0, 1.0])
    assert (coll.norm.vmin, coll.norm.vmax) == (0.0, 2.0)
    assert ax.get_xlim()[1] > 4.0  # the view grew to hold the second cell
    assert len(coll.get_paths()[0].vertices) != len(coll.get_paths()[1].vertices)


def test_polygon_mesh_by_category_fills_each_cell_with_its_class_colour():
    fig, ax = plt.subplots()
    coll = nf.polygon_mesh(
        ax, [_hexagon(0, 0), _hexagon(3, 0)], categories=["a", "b"], colors={"a": "red", "b": "blue"}
    )
    assert [tuple(c[:3]) for c in coll.get_facecolor()] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def test_polygon_mesh_takes_a_colour_per_cell_as_given():
    fig, ax = plt.subplots()
    coll = nf.polygon_mesh(ax, [_hexagon(0, 0), _hexagon(3, 0)], facecolors=["red", (0.0, 0.0, 1.0, 1.0)])
    assert [tuple(c[:3]) for c in coll.get_facecolor()] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def test_polygon_mesh_rejects_ambiguous_or_incomplete_fills():
    fig, ax = plt.subplots()
    cells = [_hexagon(0, 0)]
    with pytest.raises(ValueError, match="not several"):
        nf.polygon_mesh(ax, cells, [1.0], categories=["a"], colors={"a": "r"})
    with pytest.raises(ValueError, match="need colors"):
        nf.polygon_mesh(ax, cells, categories=["a"])
    with pytest.raises(ValueError, match="no colour"):
        nf.polygon_mesh(ax, cells, categories=["a"], colors={"b": "r"})
    with pytest.raises(ValueError, match="expected"):
        nf.polygon_mesh(ax, cells, [1.0, 2.0])


def test_outline_boxes_draws_unfilled_rectangles_at_the_given_cells():
    fig, ax = plt.subplots()
    nf.outline_boxes(ax, [(0, 1, 2, 1), (3, 0, 1, 1)], color="k", linewidth=2.0)
    assert len(ax.patches) == 2
    first = ax.patches[0]
    assert first.get_facecolor()[3] == 0.0
    assert (first.get_x(), first.get_y(), first.get_width(), first.get_height()) == (0, 1, 2, 1)


def test_one_to_one_spans_the_range_plus_its_margin_on_the_diagonal():
    fig, ax = plt.subplots()
    nf.one_to_one(ax, 0.0, 10.0)
    (line,) = ax.lines
    assert line.get_xdata().tolist() == pytest.approx([-0.5, 10.5])
    assert line.get_ydata().tolist() == line.get_xdata().tolist()
    nf.one_to_one(ax, 2.0, 2.0)
    assert ax.lines[1].get_xdata().tolist() == pytest.approx([1.9, 2.1])


def test_kde_contours_draw_for_a_cloud_and_nothing_for_a_constant():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    x = rng.normal(size=500)
    nf.kde_contours(ax, x, x + rng.normal(size=500) * 0.5, "k")
    assert len(ax.collections) >= 1
    fig2, ax2 = plt.subplots()
    nf.kde_contours(ax2, np.ones(100), rng.normal(size=100), "k")
    assert len(ax2.collections) == 0


def test_scatter_regression_reports_slope_r2_and_rmse_to_one_to_one_per_model():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    ax = nf.scatter_regression(x, {"good": x, "biased": 2.0 * x}, colors={"good": "b", "biased": "r"})
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[0].startswith("good: slope 1.00, R² 1.00, RMSE 0.00")
    assert labels[1].startswith("biased: slope 2.00, R² 1.00")
    assert ax.get_xlim() == ax.get_ylim()


def test_scatter_regression_takes_the_callers_statistic():
    x = np.linspace(0, 1, 50)
    ax = nf.scatter_regression(x, {"m": x}, colors={"m": "b"}, regress=lambda a, b: (9.0, 0.0, 0.5, 0.25))
    assert "slope 9.00, R² 0.50, RMSE 0.25" in ax.get_legend().get_texts()[0].get_text()


def test_histogram_overlays_on_one_axes_and_drops_non_finite_values():
    ax = nf.histogram([1.0, 2.0, np.nan, 3.0], bins=np.array([0.0, 2.0, 4.0]), label="a")
    nf.histogram([1.0, 1.5], ax=ax, bins=np.array([0.0, 2.0, 4.0]), label="b")
    heights = [[p.get_height() for p in c] for c in ax.containers]
    assert heights == [[1.0, 2.0], [2.0, 0.0]]


def test_colorbar_in_own_axes_is_styled_and_exclusive_with_ax():
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(4.0).reshape(2, 2))
    cax = fig.add_axes((0.1, 0.01, 0.8, 0.03))
    cb = nf.colorbar(im, cax=cax, orientation="horizontal", label="v")
    assert cb.ax is cax
    assert cb.orientation == "horizontal"
    assert cax.get_xlabel() == "v"
    assert cb.outline.get_linewidth() == 0.4
    with pytest.raises(ValueError, match="exactly one"):
        nf.colorbar(im, ax, cax=cax)
    with pytest.raises(ValueError, match="exactly one"):
        nf.colorbar(im)
    with pytest.raises(TypeError, match="without cax"):
        nf.colorbar(im, ax, spacing="proportional")
    bounded = nf.colorbar(im, cax=fig.add_axes((0.1, 0.1, 0.03, 0.5)), spacing="proportional")
    assert bounded.ax.get_ylabel() == ""


def test_legend_handle_band_carries_a_hatch():
    handle = nf.legend_handle("pending", "band", facecolor="none", edgecolor="k", hatch="///")
    assert handle.get_hatch() == "///"
