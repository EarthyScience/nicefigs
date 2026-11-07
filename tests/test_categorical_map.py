"""categorical_map: majority-vote binning, discrete legend, colour resolution.

Like the other map tests these must pass offline and without cartopy: the
projected paths are exercised only when cartopy imports, and the plain
equirectangular fallback is tested unconditionally.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import nicefigs as nf
from nicefigs.maps import categorical_map

FOUR_CLASSES = [
    "in-domain both",
    "out-of-domain both",
    "out-of-domain raw only",
    "out-of-domain SAE only",
]


def _axes():
    return plt.subplots()


def _cell_rgba(mappable, row, col):
    """Rendered colour of grid cell ``(row, col)`` of a pcolormesh mappable."""
    return tuple(mappable.to_rgba(mappable.get_array())[row, col])


def _legend_entries(ax):
    leg = ax.get_legend()
    if leg is None:
        return []
    return [(t.get_text(), tuple(h.get_facecolor())) for t, h in zip(leg.get_texts(), leg.legend_handles)]


# ------------------------------------------------------------------ majority vote


def test_binned_majority_vote_picks_the_modal_class():
    """A cell with 3 points of class 1 and 2 of class 0 must render class 1."""
    fig, ax = _axes()
    try:
        # All points fall inside the single 10 deg cell spanning lon [0, 10), lat [0, 10).
        lon = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        codes = np.array([0, 1, 1, 1, 0])
        _, m = categorical_map(
            lon,
            lat,
            codes,
            categories=["a", "b"],
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#00ff00")
    finally:
        plt.close(fig)


def test_binned_majority_vote_is_per_cell_not_global():
    """Two cells with opposite majorities must get different colours."""
    fig, ax = _axes()
    try:
        lon = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
        lat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        codes = np.array([0, 0, 1, 1, 1, 0])
        _, m = categorical_map(
            lon,
            lat,
            codes,
            categories=["a", "b"],
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#ff0000")
        assert _cell_rgba(m, 0, 1) == to_rgba("#00ff00")
    finally:
        plt.close(fig)


def test_binned_ties_resolve_to_the_earlier_category():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1, 0]),
            categories=["a", "b"],
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#ff0000")
    finally:
        plt.close(fig)


def test_binned_empty_cells_are_masked():
    fig, ax = _axes()
    try:
        lon = np.array([1.0, 2.0, 25.0])
        lat = np.array([1.0, 2.0, 1.0])
        _, m = categorical_map(
            lon,
            lat,
            np.array([0, 0, 1]),
            categories=["a", "b"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        mask = np.ma.getmaskarray(m.get_array()).ravel()
        # Three 10-deg columns are spanned; the middle one holds no point.
        assert mask.tolist() == [False, True, False]
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ legend


def test_legend_has_one_entry_per_present_category_in_order():
    fig, ax = _axes()
    try:
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        categorical_map(
            np.array([1.0, 20.0, 40.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2, 0, 3]),
            categories=FOUR_CLASSES,
            colors=colors,
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        entries = _legend_entries(ax)
        assert [label for label, _ in entries] == [FOUR_CLASSES[0], FOUR_CLASSES[2], FOUR_CLASSES[3]]
        assert [c for _, c in entries] == [to_rgba(colors[0]), to_rgba(colors[2]), to_rgba(colors[3])]
    finally:
        plt.close(fig)


def test_legend_omits_absent_categories():
    fig, ax = _axes()
    try:
        categorical_map(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([0, 0]),
            categories=["a", "b", "c"],
            kind="scatter",
            ax=ax,
        )
        assert [label for label, _ in _legend_entries(ax)] == ["a"]
    finally:
        plt.close(fig)


def test_legend_can_be_disabled_and_configured():
    fig, ax = _axes()
    try:
        categorical_map(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0]),
            categories=["a"],
            kind="scatter",
            legend=False,
            ax=ax,
        )
        assert ax.get_legend() is None
        categorical_map(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0]),
            categories=["a"],
            kind="scatter",
            legend_kw={"title": "classes", "loc": "upper right"},
            ax=ax,
        )
        assert ax.get_legend().get_title().get_text() == "classes"
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ codes & colours


def test_categories_mapping_supports_noncontiguous_codes():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([70, 70]),
            categories={10: "low", 70: "high"},
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#00ff00")
        assert [label for label, _ in _legend_entries(ax)] == ["high"]
    finally:
        plt.close(fig)


def test_string_codes_are_matched_against_category_names():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array(["forest", "forest", "water"]),
            categories=["water", "forest"],
            colors=["#0000ff", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#00ff00")
    finally:
        plt.close(fig)


def test_colors_may_be_a_dict_keyed_by_category_name_or_code():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0]),
            np.array([1.0]),
            np.array([7]),
            categories={3: "a", 7: "b"},
            colors={"a": "#ff0000", "b": "#00ff00"},
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#00ff00")
    finally:
        plt.close(fig)
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0]),
            np.array([1.0]),
            np.array([7]),
            categories={3: "a", 7: "b"},
            colors={3: "#ff0000", 7: "#0000ff"},
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#0000ff")
    finally:
        plt.close(fig)


def test_default_colors_come_from_the_house_palette():
    fig, ax = _axes()
    try:
        categorical_map(
            np.array([1.0, 20.0]),
            np.array([1.0, 1.0]),
            np.array([0, 1]),
            categories=["a", "b"],
            kind="scatter",
            ax=ax,
        )
        expected = [to_rgba(c) for c in nf.qualitative(2)]
        assert [c for _, c in _legend_entries(ax)] == expected
    finally:
        plt.close(fig)


def test_unknown_codes_are_dropped_not_recoloured():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 9, 9]),
            categories=["a", "b"],
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        # Only the single valid point survives, so the cell is class "a".
        assert _cell_rgba(m, 0, 0) == to_rgba("#ff0000")
        assert [label for label, _ in _legend_entries(ax)] == ["a"]
    finally:
        plt.close(fig)


def test_too_few_colors_raises():
    fig, ax = _axes()
    try:
        with pytest.raises(ValueError):
            categorical_map(
                np.array([1.0]),
                np.array([1.0]),
                np.array([0]),
                categories=["a", "b"],
                colors=["#ff0000"],
                kind="scatter",
                ax=ax,
            )
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ degenerate input


def test_nan_coordinates_are_dropped():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            categories=["a", "b"],
            colors=["#ff0000", "#00ff00"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert _cell_rgba(m, 0, 0) == to_rgba("#ff0000")
        assert [label for label, _ in _legend_entries(ax)] == ["a"]
    finally:
        plt.close(fig)


def test_nan_codes_are_dropped():
    fig, ax = _axes()
    try:
        categorical_map(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([0.0, np.nan]),
            categories={0.0: "a", 1.0: "b"},
            kind="scatter",
            ax=ax,
        )
        assert [label for label, _ in _legend_entries(ax)] == ["a"]
    finally:
        plt.close(fig)


def test_all_invalid_input_draws_nothing_and_does_not_raise():
    fig, ax = _axes()
    try:
        _, m = categorical_map(
            np.array([np.nan, np.nan]),
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            categories=["a", "b"],
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert m is not None
        assert ax.get_legend() is None
    finally:
        plt.close(fig)


def test_empty_input_draws_nothing_and_does_not_raise():
    fig, ax = _axes()
    try:
        _, m = categorical_map([], [], [], categories=["a", "b"], ax=ax)
        assert m is not None
        assert ax.get_legend() is None
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ kind dispatch


def test_auto_kind_scatters_sparse_clouds_and_bins_dense_ones():
    import matplotlib.collections as mcoll

    rng = np.random.default_rng(0)
    fig, ax = _axes()
    try:
        n = 200
        _, m = categorical_map(
            rng.uniform(-180, 180, n),
            rng.uniform(-60, 84, n),
            rng.integers(0, 4, n),
            categories=FOUR_CLASSES,
            ax=ax,
        )
        assert isinstance(m, mcoll.PathCollection)
    finally:
        plt.close(fig)
    fig, ax = _axes()
    try:
        n = 6000
        _, m = categorical_map(
            rng.uniform(-180, 180, n),
            rng.uniform(-60, 84, n),
            rng.integers(0, 4, n),
            categories=FOUR_CLASSES,
            bin_deg=10.0,
            ax=ax,
        )
        assert isinstance(m, mcoll.QuadMesh)
    finally:
        plt.close(fig)


def test_scatter_and_binned_agree_on_the_colour_of_a_pure_cell():
    fig, ax = _axes()
    try:
        lon, lat = np.array([1.0, 2.0]), np.array([1.0, 2.0])
        codes = np.array([1, 1])
        colors = ["#ff0000", "#00ff00"]
        _, sm = categorical_map(lon, lat, codes, categories=["a", "b"], colors=colors, kind="scatter", ax=ax)
        _, bm = categorical_map(
            lon,
            lat,
            codes,
            categories=["a", "b"],
            colors=colors,
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        point_rgba = tuple(sm.to_rgba(sm.get_array())[0])
        assert point_rgba == to_rgba("#00ff00")
        assert _cell_rgba(bm, 0, 0) == point_rgba
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ map() integration


def test_map_kind_categorical_returns_fig_ax_mappable_without_cartopy(monkeypatch):
    import nicefigs.maps as maps_mod

    monkeypatch.setattr(maps_mod, "_try_cartopy", lambda: None)
    rng = np.random.default_rng(1)
    n = 300
    fig, ax, m = nf.map(
        rng.uniform(-180, 180, n),
        rng.uniform(-60, 84, n),
        rng.integers(0, 4, n),
        kind="categorical",
        categories=FOUR_CLASSES,
    )
    try:
        assert m is not None
        assert len(_legend_entries(ax)) == 4
        # A categorical map must not grow a continuous colorbar.
        assert not [a for a in fig.axes if a is not ax]
    finally:
        plt.close(fig)


def test_map_kind_categorical_on_robinson_serves_the_threshold_overlay_use_case():
    pytest.importorskip("cartopy")
    rng = np.random.default_rng(2)
    n = 800
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-60, 84, n)
    raw, sae = rng.normal(1.0, 0.6, n), rng.normal(1.0, 0.6, n)
    codes = (raw > 1.0).astype(int) + 2 * (sae > 1.0).astype(int)
    fig, ax, m = nf.map(
        lon,
        lat,
        codes,
        kind="categorical",
        projection="robinson",
        categories={0: FOUR_CLASSES[0], 3: FOUR_CLASSES[1], 1: FOUR_CLASSES[2], 2: FOUR_CLASSES[3]},
        bin_deg=5.0,
        title="Domain agreement",
    )
    try:
        assert m is not None
        assert len(_legend_entries(ax)) == 4
    finally:
        plt.close(fig)


def test_categorical_map_accepts_a_projected_axes():
    ccrs = pytest.importorskip("cartopy.crs")
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        _, m = categorical_map(
            np.array([1.0, 20.0]),
            np.array([1.0, 10.0]),
            np.array([0, 1]),
            categories=["a", "b"],
            ccrs=ccrs,
            kind="binned",
            bin_deg=10.0,
            ax=ax,
        )
        assert m is not None
        assert len(_legend_entries(ax)) == 2
    finally:
        plt.close(fig)
