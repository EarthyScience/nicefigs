"""style.py: themes, scaling ratios, rc(), style() context manager."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import nicefigs as nf
from nicefigs.style import THEMES, Theme, get_theme, set_theme, style


def test_get_theme_default_is_paper():
    th = get_theme(None)
    assert th.name == "paper"


def test_get_theme_by_name():
    th = get_theme("talk")
    assert th is THEMES["talk"]


def test_get_theme_passthrough_theme_instance():
    custom = Theme.from_base(11.0, name="custom")
    assert get_theme(custom) is custom


def test_get_theme_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_theme("not-a-theme")


def test_from_base_scaling_ratios():
    base = 10.0
    th = Theme.from_base(base)
    assert th.fs_body == pytest.approx(base)
    assert th.fs_panel_label == pytest.approx(base * 1.2)
    assert th.fs_title == pytest.approx(base * 1.08)
    assert th.fs_panel_title == pytest.approx(base * 1.1)
    assert th.fs_axis_label == pytest.approx(base * 1.00)
    assert th.fs_tick == pytest.approx(base * 0.85)
    assert th.fs_legend == pytest.approx(base * 0.92)
    assert th.fs_annot == pytest.approx(base * 0.85)


def test_from_base_overrides_win():
    th = Theme.from_base(10.0, fs_title=99.0)
    assert th.fs_title == 99.0


def test_scaled_multiplies_all_font_sizes():
    th = Theme.from_base(10.0)
    th2 = th.scaled(2.0)
    assert th2.base == pytest.approx(th.base * 2.0)
    assert th2.fs_body == pytest.approx(th.fs_body * 2.0)
    assert th2.fs_panel_label == pytest.approx(th.fs_panel_label * 2.0)
    assert th2.fs_tick == pytest.approx(th.fs_tick * 2.0)
    # Non-font fields are untouched.
    assert th2.lw_axes == th.lw_axes


def test_with_overrides_a_copy():
    th = get_theme("paper")
    th2 = th.with_(lw_line=5.0)
    assert th2.lw_line == 5.0
    assert th.lw_line != 5.0


def test_theme_presets_exist():
    for name in ("paper", "paper-large", "talk", "poster"):
        assert name in THEMES


def test_talk_and_poster_are_bigger_than_paper():
    paper = THEMES["paper"]
    talk = THEMES["talk"]
    poster = THEMES["poster"]
    assert talk.base > paper.base
    assert poster.base > talk.base


def test_rc_returns_dict_with_valid_matplotlib_keys():
    th = get_theme("paper")
    rc = th.rc()
    assert isinstance(rc, dict)
    assert rc["font.size"] == th.fs_body
    assert rc["axes.labelsize"] == th.fs_axis_label
    # Every key should be a real, settable rcParams key (raises if not); spot
    # check a representative subset of scalar values round-trip correctly.
    import matplotlib as mpl

    with mpl.rc_context(rc):
        for k in rc:
            assert k in mpl.rcParams
        for k in (
            "font.size",
            "axes.labelsize",
            "xtick.labelsize",
            "axes.linewidth",
            "lines.linewidth",
            "figure.dpi",
            "savefig.dpi",
            "text.color",
        ):
            assert mpl.rcParams[k] == rc[k]
        assert list(mpl.rcParams["font.sans-serif"]) == list(rc["font.sans-serif"])


def test_set_theme_updates_active_and_rcparams():
    prev = get_theme()
    try:
        resolved = set_theme("talk")
        assert resolved.name == "talk"
        assert get_theme() is resolved
        assert plt.rcParams["font.size"] == resolved.fs_body
    finally:
        set_theme(prev)


def test_style_context_manager_restores_rcparams_on_exit():
    set_theme("paper")
    before = plt.rcParams["font.size"]
    with style("talk"):
        assert plt.rcParams["font.size"] == THEMES["talk"].fs_body
        assert get_theme().name == "talk"
    after = plt.rcParams["font.size"]
    assert after == before
    assert get_theme().name == "paper"


def test_style_context_manager_restores_on_exception():
    set_theme("paper")
    before = plt.rcParams["font.size"]
    with pytest.raises(ValueError):
        with style("talk"):
            assert plt.rcParams["font.size"] == THEMES["talk"].fs_body
            raise ValueError("boom")
    assert plt.rcParams["font.size"] == before
    assert get_theme().name == "paper"


def test_style_context_manager_accepts_rc_overrides():
    with style("paper", **{"axes.grid": True}):
        assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.grid"] is False


def test_style_unknown_theme_raises():
    with pytest.raises(KeyError):
        with style("nope"):
            pass


def test_colors_dict_has_expected_semantic_keys():
    for key in ("primary", "accent", "good", "bad", "land", "ocean"):
        assert key in nf.COLORS


def test_palettes_are_nonempty_hex_lists():
    for pal in (nf.PALETTE_BRIGHT, nf.PALETTE_MUTED, nf.PALETTE_HIGH_CONTRAST):
        assert len(pal) > 0
        for c in pal:
            assert c.startswith("#")
