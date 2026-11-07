"""Contract tests for the 2D-colormap, ternary, image and map helpers."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs.colormaps2d import BIVARIATE_CORNERS, normalize01
from nicefigs.ternary import barycentric_to_xy


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_normalize01_maps_range_to_unit_interval():
    x = np.arange(101.0)
    scaled, (lo, hi) = normalize01(x, method="linear")
    assert (lo, hi) == (0.0, 100.0)
    np.testing.assert_allclose(scaled, x / 100)
    q, (qlo, qhi) = normalize01(x)  # quantile: 5th/95th percentiles map to 0/1
    assert (qlo, qhi) == (5.0, 95.0)
    assert q.min() == 0.0 and q.max() == 1.0


def test_normalize01_all_nan_is_safe():
    scaled, rng = normalize01([np.nan, np.nan])
    assert rng == (0.0, 1.0) and scaled.shape == (2,)


def test_bivariate_cmap_hits_corner_colours():
    rgb = nf.bivariate_cmap([0, 1, 0, 1], [0, 0, 1, 1])
    expected = [mcolors.to_rgb(BIVARIATE_CORNERS[k]) for k in ("ll", "hl", "lh", "hh")]
    np.testing.assert_allclose(rgb, expected)


def test_hsv_cmap_keeps_value_fixed():
    rgb = nf.hsv_cmap(np.linspace(0, 1, 7), np.linspace(1, 0, 7), value=0.7)
    np.testing.assert_allclose(mcolors.rgb_to_hsv(rgb)[:, 2], 0.7, atol=1e-6)


def test_flat_cmap_transpose_swaps_axes():
    x, y = np.array([0.1, 0.9]), np.array([0.8, 0.3])
    np.testing.assert_allclose(nf.flat_cmap(x, y, transpose=True), nf.flat_cmap(y, x))


def test_flat_legend_accepts_rotation(ax):
    nf.flat_legend(ax, x_label="a", y_label="b", rotation_deg=-75, transpose=True)
    assert len(ax.images) == 1


def test_barycentric_vertices():
    xy = barycentric_to_xy(np.eye(3))
    np.testing.assert_allclose(xy, [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])


def test_ternary_color_pure_components_are_corner_colours():
    rgb = nf.ternary_color(np.eye(3) * 4.0, corners="bright", gamma=1.0)  # unnormalised input
    expected = [mcolors.to_rgb(c) for c in nf.TERNARY_PALETTES["bright"]]
    np.testing.assert_allclose(rgb, expected)


def test_ternary_scatter_value_mode_returns_mappable():
    comp = np.random.default_rng(0).dirichlet([1, 1, 1], 20)
    ax, mappable = nf.ternary_scatter(comp, comp[:, 0], color_by="value")
    try:
        assert mappable is not None
        assert np.asarray(mappable.get_array()).shape == (20,)
    finally:
        plt.close(ax.figure)


def test_image_rgb_returns_none_and_intensity_returns_mappable(ax):
    assert nf.image(np.zeros((4, 4, 3)), ax=ax) == (ax, None)
    _, m = nf.image(np.arange(16.0).reshape(4, 4), robust=True, ax=ax)
    assert m is not None and m.get_clim()[0] < m.get_clim()[1]


def test_heatmap_annotates_finite_cells_only(ax):
    mat = np.array([[1.0, np.nan], [3.0, 4.0]])
    nf.heatmap(mat, annotate=True, row_labels=["a", "b"], col_labels=["c", "d"], ax=ax)
    assert sorted(t.get_text() for t in ax.texts) == ["1.00", "3.00", "4.00"]


def test_make_map_axes_rejects_unknown_projection():
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="projection"):
            nf.make_map_axes(fig, projection="not-a-projection")
    finally:
        plt.close(fig)


def test_scatter_map_drops_nonfinite_values(ax):
    _, sc = nf.scatter_map([0, 10, 20], [0, 5, 10], [1.0, np.nan, 3.0], ax=ax)
    assert sc.get_offsets().shape == (2, 2)


def test_map_does_not_mutate_cbar_kwargs():
    lon, lat = np.meshgrid(np.linspace(-170, 170, 6), np.linspace(-50, 70, 4))
    field = np.arange(lon.size, dtype=float).reshape(lon.shape)
    kwargs = {"location": "bottom"}
    fig, _, m = nf.map(
        lon[0], lat[:, 0], field, cbar=True, cbar_kwargs=kwargs, inset_hist=False, land=False, coastline=False
    )
    try:
        assert m is not None
        assert kwargs == {"location": "bottom"}
    finally:
        plt.close(fig)
