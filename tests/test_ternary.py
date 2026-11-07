"""ternary.py: composition colour encoding, ternary scatter/map/image/legend."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd
from nicefigs.ternary import normalize_composition


def test_normalize_composition_rows_sum_to_one():
    comp = np.array([[1.0, 1.0, 2.0], [0.0, 0.0, 5.0]])
    out = normalize_composition(comp)
    sums = out.sum(axis=-1)
    assert sums == pytest.approx([1.0, 1.0])


def test_normalize_composition_zero_row_is_nan():
    comp = np.array([[0.0, 0.0, 0.0]])
    out = normalize_composition(comp)
    assert np.all(np.isnan(out))


def test_ternary_color_pure_corner_returns_corner_color():
    from matplotlib.colors import to_rgb

    corners = ("#d73027", "#1a9850", "#4575b4")
    # pure "A" corner
    comp = np.array([[1.0, 0.0, 0.0]])
    rgb = nf.ternary_color(comp, corners=corners, gamma=1.0)
    assert rgb.shape == (1, 3)
    assert rgb[0] == pytest.approx(to_rgb(corners[0]), abs=1e-6)


def test_ternary_named_palette_resolves():
    from matplotlib.colors import to_rgb

    from nicefigs.ternary import resolve_corners

    assert "bright" in nf.TERNARY_PALETTES
    # a named palette flows through ternary_color (pure corner → that colour)
    rgb = nf.ternary_color([[1.0, 0.0, 0.0]], corners="bright", gamma=1.0)
    assert rgb[0] == pytest.approx(to_rgb(nf.TERNARY_PALETTES["bright"][0]), abs=1e-6)
    # explicit 3-tuple still works; wrong length / unknown name raise
    assert resolve_corners(("#000000", "#ffffff", "#ff0000"))[1] == "#ffffff"
    with pytest.raises(KeyError):
        resolve_corners("not-a-palette")
    with pytest.raises(ValueError):
        resolve_corners(("#000", "#fff"))


def test_ternary_scatter_accepts_palette_name():
    comp, _ = npd.ternary_points(n=80)
    ax, _ = nf.ternary_scatter(comp, corners="bright", labels=("x", "y", "z"))
    try:
        assert len(ax.collections) >= 1
    finally:
        plt.close(ax.figure)


def test_ternary_color_shape_and_range_for_batch():
    comp, values = npd.ternary_points(n=100)
    rgb = nf.ternary_color(comp)
    assert rgb.shape == (100, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_ternary_color_image_shape():
    comp_img = npd.ternary_points(n=64)[0].reshape(8, 8, 3)
    rgb = nf.ternary_color(comp_img)
    assert rgb.shape == (8, 8, 3)


def test_ternary_scatter_composition_mode():
    comp, values = npd.ternary_points(n=150)
    ax, sc = nf.ternary_scatter(comp, values)
    try:
        assert sc is None  # composition mode returns no separate mappable
        assert len(ax.collections) >= 1
    finally:
        plt.close(ax.figure)


def test_ternary_scatter_value_mode_returns_mappable():
    comp, values = npd.ternary_points(n=150)
    ax, sc = nf.ternary_scatter(comp, values, color_by="value")
    try:
        assert sc is not None
    finally:
        plt.close(ax.figure)


def test_ternary_map_smoke():
    comp, values = npd.ternary_points(n=100)
    lon = np.random.default_rng(0).uniform(-180, 180, 100)
    lat = np.random.default_rng(1).uniform(-90, 90, 100)
    fig, ax = nf.ternary_map(lon, lat, comp)
    try:
        assert fig is not None
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_ternary_image_smoke():
    comp_img, _ = npd.ternary_points(n=64)
    comp_img = comp_img.reshape(8, 8, 3)
    ax = nf.ternary_image(comp_img, title="ternary")
    try:
        assert len(ax.images) == 1
    finally:
        plt.close(ax.figure)


def test_ternary_legend_smoke():
    ax = nf.ternary_legend()
    try:
        assert len(ax.images) == 1
    finally:
        plt.close(ax.figure)


def test_ternary_axes_sets_equal_aspect_and_limits():
    ax = nf.ternary_axes()
    try:
        assert ax.get_aspect() in ("equal", 1.0)
        xlim = ax.get_xlim()
        assert xlim[0] < 0 < 1 < xlim[1]
    finally:
        plt.close(ax.figure)
