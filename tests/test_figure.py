"""figure.py: figure_size math, figure()/subplots() physical sizing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import nicefigs as nf
from nicefigs.figure import CANVASES, DEFAULT_ASPECT, WIDTHS, figure_size

MM = 1.0 / 25.4


def test_named_width_presets_exact_values():
    assert figure_size("single") == pytest.approx((89 * MM, 89 * MM * DEFAULT_ASPECT))
    assert figure_size("full") == pytest.approx((5.5, 5.5 * DEFAULT_ASPECT))
    assert figure_size("double") == pytest.approx((183 * MM, 183 * MM * DEFAULT_ASPECT))


def test_all_width_presets_resolve():
    for name, w in WIDTHS.items():
        width, height = figure_size(name)
        assert width == pytest.approx(w)
        assert height == pytest.approx(w * DEFAULT_ASPECT)


def test_canvas_presets_return_exact_wh():
    for name, (w, h) in CANVASES.items():
        assert figure_size(name) == pytest.approx((w, h))


def test_canvas_preset_with_explicit_aspect_uses_only_width():
    # When aspect/height/panel_aspect given alongside a canvas name, it should
    # be treated like a plain width preset (first CANVASES value), not the
    # (w, h) canvas shortcut.
    w, h = figure_size("slide-16-9", aspect=0.5)
    assert w == pytest.approx(CANVASES["slide-16-9"][0])
    assert h == pytest.approx(w * 0.5)


def test_width_and_aspect_solves_height():
    w, h = figure_size(width=10.0, aspect=0.5)
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(5.0)


def test_width_only_uses_default_aspect():
    w, h = figure_size(width=10.0)
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(10.0 * DEFAULT_ASPECT)


def test_height_and_aspect_solves_width():
    w, h = figure_size(height=5.0, aspect=0.5)
    assert h == pytest.approx(5.0)
    assert w == pytest.approx(10.0)


def test_height_only_uses_default_aspect():
    w, h = figure_size(height=6.6)
    assert h == pytest.approx(6.6)
    assert w == pytest.approx(6.6 / DEFAULT_ASPECT)


def test_width_and_height_both_given_ignores_aspect():
    w, h = figure_size(width=4.0, height=3.0, aspect=99.0)
    assert (w, h) == (4.0, 3.0)


def test_nothing_given_returns_sensible_default():
    w, h = figure_size()
    assert w == pytest.approx(WIDTHS["half"])
    assert h == pytest.approx(WIDTHS["half"] * DEFAULT_ASPECT)


def test_panel_aspect_shapes_panels():
    # panel_aspect takes precedence over aspect, and scales with nrows/ncols.
    w, h = figure_size(width=10.0, nrows=2, ncols=4, panel_aspect=1.0)
    expected_aspect = (2 / 4) * 1.0
    assert h == pytest.approx(10.0 * expected_aspect)


def test_panel_aspect_overrides_aspect():
    w1, h1 = figure_size(width=10.0, aspect=0.9, nrows=1, ncols=1, panel_aspect=0.3)
    w2, h2 = figure_size(width=10.0, nrows=1, ncols=1, panel_aspect=0.3)
    assert h1 == pytest.approx(h2)


def test_unknown_width_preset_raises():
    with pytest.raises(KeyError):
        figure_size(width="not-a-real-preset")


def test_figure_matches_requested_figsize():
    fig = nf.figure(width=4.0, aspect=0.5)
    try:
        assert fig.get_size_inches() == pytest.approx((4.0, 2.0), abs=1e-6)
    finally:
        plt.close(fig)


def test_figure_named_width():
    fig = nf.figure(width="single", aspect=1.0)
    try:
        w, h = fig.get_size_inches()
        assert w == pytest.approx(89 * MM, abs=1e-6)
        assert h == pytest.approx(89 * MM, abs=1e-6)
    finally:
        plt.close(fig)


def test_subplots_matches_requested_figsize():
    fig, axes = nf.subplots(2, 2, width=8.0, panel_aspect=1.0)
    try:
        expected_h = 8.0 * (2 / 2) * 1.0
        assert fig.get_size_inches() == pytest.approx((8.0, expected_h), abs=1e-6)
        assert axes.shape == (2, 2)
    finally:
        plt.close(fig)


def test_subplots_routes_wspace_hspace_into_gridspec():
    fig, axes = nf.subplots(1, 2, width=6.0, wspace=0.5)
    try:
        assert axes.shape == (2,)
    finally:
        plt.close(fig)


def test_save_writes_single_and_multi_format(tmp_path):
    fig = nf.figure(width=2.0, aspect=1.0)
    try:
        (out,) = nf.save(fig, tmp_path / "fig", formats=("png",))
        assert out.exists()
        assert out.suffix == ".png"

        outs = nf.save(fig, tmp_path / "fig2", formats=("png", "pdf"))
        assert [p.suffix for p in outs] == [".png", ".pdf"]
        for p in outs:
            assert p.exists()
    finally:
        plt.close(fig)


def test_save_default_format(tmp_path):
    # With no `formats`, save() passes the path straight to fig.savefig(), so
    # the path needs its own extension (matches the documented usage pattern).
    fig = nf.figure(width=2.0, aspect=1.0)
    try:
        (out,) = nf.save(fig, tmp_path / "sub" / "fig.png")
        assert out.exists()
    finally:
        plt.close(fig)


def test_save_close_option(tmp_path):
    fig = nf.figure(width=2.0, aspect=1.0)
    nf.save(fig, tmp_path / "fig", close=True)
    assert not plt.fignum_exists(fig.number)
