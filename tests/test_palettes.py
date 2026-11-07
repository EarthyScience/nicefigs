"""palettes.py: cmap resolution, categorical palettes, limits, elevation cmap."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest

from nicefigs.palettes import (
    CATEGORICAL,
    DEFAULTS,
    SCORECARD_DIVERGING_BOUNDARIES,
    categorical,
    cmap,
    diverging_norm,
    elevation_cmap,
    qualitative,
    robust_limits,
    scorecard_diverging_cmap,
    symmetric_limits,
)


def test_cmap_resolves_house_alias():
    cm = cmap("spectral_r")
    assert isinstance(cm, mcolors.Colormap)
    assert cm.name.lower() == "spectral_r"


def test_cmap_resolves_house_default_kind():
    cm = cmap("diverging")
    assert cm.name == DEFAULTS["diverging"]


def test_cmap_resolves_matplotlib_name_passthrough():
    cm = cmap("viridis")
    assert cm.name == "viridis"


def test_cmap_passthrough_colormap_instance():
    src = mpl.colormaps["magma"]
    cm = cmap(src)
    assert cm is src


def test_cmap_resamples_to_n_levels():
    cm = cmap("inferno", n=5)
    assert cm.N == 5


def test_cmap_unknown_name_raises():
    with pytest.raises(KeyError):
        cmap("definitely-not-a-real-colormap-xyz")


def test_categorical_products_names_match_colors():
    for name in CATEGORICAL:
        cat = categorical(name)
        assert set(cat.names) == set(cat.colors)
        assert cat.codes == sorted(cat.colors)


def test_categorical_unknown_raises():
    with pytest.raises(KeyError):
        categorical("not-a-product")


def test_categorical_cmap_norm_maps_each_code():
    cat = categorical("esa_worldcover")
    cm, norm = cat.cmap_norm()
    assert isinstance(cm, mcolors.ListedColormap)
    assert cm.N == len(cat.codes)
    for code in cat.codes:
        rgba = cm(norm(code))
        expected = mcolors.to_rgba(cat.colors[code])
        assert rgba == pytest.approx(expected)


def test_categorical_to_rgb_shape_and_values():
    cat = categorical("esri_lulc")
    codes = cat.codes
    arr = np.array([[codes[0], codes[1]], [codes[2], 9999]])
    rgb = cat.to_rgb(arr)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0] == pytest.approx(mcolors.to_rgb(cat.colors[codes[0]]), abs=1e-6)
    # unknown code falls back to default grey
    assert rgb[1, 1] == pytest.approx(mcolors.to_rgb("#dddddd"), abs=1e-6)


def test_categorical_legend_handles_count():
    cat = categorical("modis_igbp")
    handles = cat.legend_handles()
    assert len(handles) == len(cat.codes)
    subset = cat.codes[:3]
    handles_subset = cat.legend_handles(subset)
    assert len(handles_subset) == 3


def test_elevation_cmap_returns_cmap_and_norm_with_sea_level_break():
    cm, norm = elevation_cmap(sea_level=0.0, vmin=-8000.0, vmax=6000.0)
    assert isinstance(cm, mcolors.LinearSegmentedColormap)
    assert isinstance(norm, mcolors.Normalize)
    assert norm.vmin == -8000.0
    assert norm.vmax == 6000.0
    # at sea level, colour should be at the ocean/land break fraction
    break_frac = (0.0 - (-8000.0)) / (6000.0 - (-8000.0))
    assert norm(0.0) == pytest.approx(break_frac)


def test_elevation_cmap_ocean_and_land_differ():
    cm, norm = elevation_cmap()
    deep_ocean = cm(norm(-7000.0))
    high_land = cm(norm(5000.0))
    assert deep_ocean != high_land


def test_robust_limits_basic():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 10000)
    lo, hi = robust_limits(vals)
    assert lo < 0 < hi
    assert lo == pytest.approx(np.percentile(vals, 2), abs=1e-6)
    assert hi == pytest.approx(np.percentile(vals, 98), abs=1e-6)


def test_robust_limits_all_nan_returns_default():
    lo, hi = robust_limits(np.full(10, np.nan))
    assert (lo, hi) == (0.0, 1.0)


def test_robust_limits_empty_returns_default():
    lo, hi = robust_limits(np.array([]))
    assert (lo, hi) == (0.0, 1.0)


def test_robust_limits_constant_values_widens_slightly():
    lo, hi = robust_limits(np.full(20, 5.0))
    assert lo == 5.0
    assert hi > lo


def test_symmetric_limits_basic():
    vals = np.array([-10.0, -2.0, 3.0, 9.0])
    lo, hi = symmetric_limits(vals, percentile=100.0)
    assert lo == -hi
    assert hi == pytest.approx(10.0)


def test_symmetric_limits_all_nan_returns_default():
    lo, hi = symmetric_limits(np.full(5, np.nan))
    assert (lo, hi) == (-1.0, 1.0)


def test_symmetric_limits_empty_returns_default():
    lo, hi = symmetric_limits(np.array([]))
    assert (lo, hi) == (-1.0, 1.0)


def test_diverging_norm_centers_correctly():
    norm = diverging_norm(-5.0, 5.0, center=0.0)
    assert isinstance(norm, mcolors.TwoSlopeNorm)
    assert norm.vcenter == 0.0
    assert norm(0.0) == pytest.approx(0.5)


def test_diverging_norm_handles_center_outside_range():
    # center must be strictly between vmin and vmax; function should adjust.
    norm = diverging_norm(1.0, 2.0, center=0.0)
    assert norm.vmin <= 0.0
    assert norm.vmax >= 0.0


def test_qualitative_returns_requested_count():
    cols = qualitative(4)
    assert len(cols) == 4


def test_qualitative_wraps_when_exceeding_palette_length():
    from nicefigs.style import PALETTE_HIGH_CONTRAST

    n = len(PALETTE_HIGH_CONTRAST) + 3
    cols = qualitative(n, palette="high_contrast")
    assert len(cols) == n
    assert cols[0] == cols[len(PALETTE_HIGH_CONTRAST)]


# ------------------------------------------------------- scorecard diverging
def test_scorecard_diverging_boundaries_match_spec():
    assert SCORECARD_DIVERGING_BOUNDARIES == [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]


def test_scorecard_diverging_cmap_resolves_via_registry():
    cm = cmap("scorecard_diverging")
    assert isinstance(cm, mcolors.ListedColormap)
    assert cm.N == 11


def test_scorecard_diverging_cmap_returns_cmap_and_boundary_norm():
    cm, norm = scorecard_diverging_cmap()
    assert isinstance(cm, mcolors.ListedColormap)
    assert isinstance(norm, mcolors.BoundaryNorm)
    assert list(norm.boundaries) == SCORECARD_DIVERGING_BOUNDARIES


def test_scorecard_diverging_cmap_bands_value():
    # -35 falls in the [-50, -20) band → the first (most negative) band colour.
    cm, norm = scorecard_diverging_cmap()
    assert cm(norm(-35)) == pytest.approx(mcolors.to_rgba("#437AB5"))


def test_scorecard_diverging_cmap_over_extend():
    cm, norm = scorecard_diverging_cmap()
    assert cm(norm(60)) == pytest.approx(mcolors.to_rgba("#9C2520"))


def test_scorecard_diverging_cmap_under_extend():
    cm, norm = scorecard_diverging_cmap()
    assert cm(norm(-60)) == pytest.approx(mcolors.to_rgba("#26549A"))


def test_scorecard_diverging_cmap_extend_colors_distinct_from_bands():
    cm, norm = scorecard_diverging_cmap()
    assert cm(norm(-60)) != cm(norm(-35))  # under colour != outermost band
    assert cm(norm(60)) != cm(norm(35))  # over colour != outermost band


def test_qualitative_unknown_palette_raises():
    with pytest.raises(KeyError, match="Unknown palette"):
        qualitative(3, "no-such-palette")
