"""artists.py: MetBrewer / MoMAColors / Hockney palettes and colormaps."""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

import nicefigs as nf
from nicefigs.artists import HOCKNEY_CANYON, METBREWER, MOMA


def test_hex_values_are_valid():
    for name, cols in {**METBREWER, **MOMA, "hockney": HOCKNEY_CANYON}.items():
        assert len(cols) >= 3, name
        for c in cols:
            mcolors.to_rgb(c)  # raises if invalid


def test_artist_palette_lookup_truncate_cycle_reverse():
    full = nf.artist_palette("hiroshige")
    assert len(full) == 10
    assert nf.artist_palette("hiroshige", 3) == full[:3]
    assert nf.artist_palette("hiroshige", reverse=True) == full[::-1]
    # cycles beyond length
    assert len(nf.artist_palette("andri", 10)) == 10
    with pytest.raises(KeyError):
        nf.artist_palette("not-an-artist")


def test_hockney_is_a_qualitative_palette():
    cols = nf.qualitative(5, "hockney")
    assert cols == HOCKNEY_CANYON[:5]


def test_nice_is_the_default_cycle():
    from nicefigs.artists import NICE

    # nice leads with teal, then orange, then purple
    assert NICE[:3] == ["#25c5a6", "#ef5a1c", "#9a4fbc"]
    # it is the default for qualitative() and the theme colour cycle
    assert nf.qualitative(3) == NICE[:3]
    assert list(nf.get_theme("paper").palette)[:3] == NICE[:3]


def test_artist_cmaps_registered_and_resolvable():
    for name in ("hockney_canyon", "hiroshige", "hokusai2", "ernst", "flash"):
        cm = nf.cmap(name)
        assert isinstance(cm, mcolors.Colormap)
        assert nf.cmap(f"{name}_r") is not None
    # artist_cmap builds one for any palette
    assert nf.artist_cmap("klimt").name == "klimt"


def test_nice_colors_and_light_cmap():
    from nicefigs.artists import NICE_COLORS, light_cmap

    assert NICE_COLORS["teal"] == "#25c5a6"
    assert NICE_COLORS["orange"] == "#ef5a1c"
    cm = light_cmap(NICE_COLORS["teal"], "vt")
    # white at the low end, teal-ish in the middle
    assert min(cm(0.0)[:3]) > 0.98
    r, g, b, _ = cm(0.5)
    assert g > r and b > r  # teal: green+blue dominate red
    # registered as a resolvable colormap
    assert nf.cmap("nice_teal").name == "nice_teal"


def test_agreement_uses_nice_theme_by_default():
    import matplotlib.pyplot as plt

    from nicefigs import data as npd
    from nicefigs.artists import NICE_COLORS

    obs, pred = npd.xy_agreement(n=300)
    ax, _ = nf.agreement(obs, pred, density="hist")  # hist avoids kde singularities
    try:
        fit = [ln for ln in ax.lines if ln.get_label() == "fit"]
        assert fit and mcolors.to_hex(fit[0].get_color()) == NICE_COLORS["orange"]
    finally:
        plt.close(ax.figure)


def test_hockney_cmap_spans_blue_to_warm():
    cm = nf.cmap("hockney_canyon")
    lo = cm(0.0)
    hi = cm(1.0)
    # low end is bluer than red, high end is warmer (redder) than blue
    assert lo[2] > lo[0]
    assert hi[0] > hi[2]
