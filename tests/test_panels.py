"""panels.py: grid layout, panel labelling, titles, despine."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf


def test_grid_returns_expected_shape():
    fig, axes = nf.grid(2, 3, width=8.0, panel_aspect=1.0)
    try:
        assert axes.shape == (2, 3)
        assert all(ax is not None for ax in axes.ravel())
    finally:
        plt.close(fig)


def test_panel_label_lowercase_default():
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a")
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "a"
    finally:
        plt.close(fig)


def test_panel_label_parens_and_title():
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "b", title="My panel", parens=True)
        texts = [t.get_text() for t in ax.texts]
        assert "(b)" in texts
        assert "My panel" in texts
    finally:
        plt.close(fig)


def test_label_panels_sequential_letters():
    fig, axes = plt.subplots(1, 3)
    try:
        nf.label_panels(axes)
        labels = [ax.texts[0].get_text() for ax in axes]
        assert labels == ["a", "b", "c"]
    finally:
        plt.close(fig)


def test_label_panels_upper_and_start_offset():
    fig, axes = plt.subplots(1, 2)
    try:
        nf.label_panels(axes, upper=True, start=2)
        labels = [ax.texts[0].get_text() for ax in axes]
        assert labels == ["C", "D"]
    finally:
        plt.close(fig)


def test_label_panels_skips_none():
    fig, axes = plt.subplots(1, 3)
    arr = np.array([axes[0], None, axes[2]], dtype=object)
    try:
        nf.label_panels(arr)
        assert len(axes[0].texts) == 1
        assert len(axes[1].texts) == 0
        assert len(axes[2].texts) == 1
        assert axes[0].texts[0].get_text() == "a"
        assert axes[2].texts[0].get_text() == "b"
    finally:
        plt.close(fig)


def test_label_panels_with_titles():
    fig, axes = plt.subplots(1, 2)
    try:
        nf.label_panels(axes, titles=["First", "Second"])
        assert any(t.get_text() == "First" for t in axes[0].texts)
        assert any(t.get_text() == "Second" for t in axes[1].texts)
    finally:
        plt.close(fig)


def _anchor_px(t):
    """Display coordinates of a text's anchor (its baseline start for va="baseline")."""
    return t.get_transform().transform(t.get_position())


def test_panel_label_default_sits_flush_left_above_the_axes():
    """Bold lowercase letter, left edge on the axes' left edge, baseline half a label-size above the top."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a")
        assert [t.get_text() for t in ax.texts] == ["a"]
        t = ax.texts[0]
        assert t.get_fontweight() == "bold"
        assert t.get_fontsize() == pytest.approx(nf.get_theme().fs_panel_label)
        x_px, y_px = _anchor_px(t)
        box = ax.get_window_extent()
        pad_px = 0.5 * nf.get_theme().fs_panel_label * fig.dpi / 72.0
        assert x_px == pytest.approx(box.x0, abs=0.5)
        assert y_px == pytest.approx(box.y1 + pad_px, abs=0.5)
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_label_panels_default_unaffected_by_new_kwargs():
    fig, axes = plt.subplots(1, 3)
    try:
        nf.label_panels(axes)
        for ax, letter in zip(axes, "abc"):
            assert len(ax.texts) == 1
            assert ax.texts[0].get_text() == letter
            assert ax.texts[0].get_fontweight() == "bold"
    finally:
        plt.close(fig)


def test_panel_label_composes_letter_and_title_one_line():
    """Letter and title share a baseline; the title starts after the letter, in normal weight."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="Model")
        letter_text, title_text = ax.texts
        assert (letter_text.get_text(), title_text.get_text()) == ("a", "Model")
        assert letter_text.get_fontweight() == "bold"
        assert title_text.get_fontweight() == "normal"
        assert letter_text.get_fontsize() > title_text.get_fontsize()
        ly, ty = _anchor_px(letter_text)[1], _anchor_px(title_text)[1]
        assert ty == pytest.approx(ly, abs=0.5)
        renderer = fig.canvas.get_renderer()
        assert title_text.get_window_extent(renderer).x0 > letter_text.get_window_extent(renderer).x1
    finally:
        plt.close(fig)


def test_panel_label_use_ax_title_composes_and_clears_mpl_title():
    fig, ax = plt.subplots()
    try:
        ax.set_title("Model")
        nf.panel_label(ax, "a", use_ax_title=True)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "Model"]
        assert ax.texts[0].get_fontweight() == "bold"
        assert ax.texts[1].get_fontweight() == "normal"
        # matplotlib's own title is cleared so it isn't drawn a second time,
        # at its own (misaligned) default position.
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_panel_label_use_ax_title_picks_up_loc_left_title():
    """``clean_title``/``hovmoller`` set loc="left", not the get_title() default."""
    fig, ax = plt.subplots()
    try:
        ax.set_title("Model", loc="left")
        nf.panel_label(ax, "a", use_ax_title=True)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "Model"]
        assert ax.get_title(loc="left") == ""
    finally:
        plt.close(fig)


def test_panel_label_use_ax_title_noop_without_existing_title():
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", use_ax_title=True)
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "a"
    finally:
        plt.close(fig)


def test_panel_label_dedupes_title_when_dedupe_true():
    """When explicitly enabled, dedupe strips a leading duplicate of the label."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="a Model", dedupe=True)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "Model"]
    finally:
        plt.close(fig)


def test_panel_label_dedupe_does_not_touch_coincidental_prefix():
    """A title merely starting with the same character is left untouched."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="Aircraft profile")
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "Aircraft profile"]
    finally:
        plt.close(fig)


def test_panel_label_dedupe_false_keeps_duplicate():
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="a Model", dedupe=False)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "a Model"]
    finally:
        plt.close(fig)


def test_panel_label_default_no_dedupe_keeps_title_unchanged():
    """Default behavior (dedupe=False): titles starting with the label are unchanged."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="a Model")
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "a Model"]
    finally:
        plt.close(fig)


def test_panel_label_default_preserves_a_priori():
    """Verify the 'a priori' false-positive case: with dedupe=False (default),
    'a priori flux' is rendered fully, not stripped to 'priori flux'."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="a priori flux")
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "a priori flux"]
    finally:
        plt.close(fig)


def test_panel_label_default_preserves_a_posteriori():
    """Verify another common false-positive: 'a posteriori' is not stripped."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "a", title="a posteriori estimate")
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["a", "a posteriori estimate"]
    finally:
        plt.close(fig)


def test_panel_label_default_preserves_b_spline():
    """Verify the 'b spline' false-positive case on panel 'b'."""
    fig, ax = plt.subplots()
    try:
        nf.panel_label(ax, "b", title="b spline fit")
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["b", "b spline fit"]
    finally:
        plt.close(fig)


def test_label_panels_use_ax_title_per_panel_fallback():
    fig, axes = plt.subplots(1, 3)
    try:
        axes[0].set_title("Model")
        axes[1].set_title("Reference")
        axes[2].set_title("Difference")
        nf.label_panels(axes, use_ax_title=True)
        expected = [("a", "Model"), ("b", "Reference"), ("c", "Difference")]
        for ax, (letter, title) in zip(axes, expected):
            texts = [t.get_text() for t in ax.texts]
            assert texts == [letter, title]
            assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_panel_label_default_placement_within_axes_bounds_and_consistent():
    fig, axes = plt.subplots(1, 3)
    try:
        nf.label_panels(axes, titles=["First", "Second", "Third"])
        positions = [ax.texts[0].get_position() for ax in axes]
        # Same anchor on every panel — callers never hand-tune coordinates.
        assert len(set(positions)) == 1
        x, y = positions[0]
        assert -0.2 <= x <= 0.2
        assert 0.8 <= y <= 1.2
    finally:
        plt.close(fig)


def test_clean_title_sets_left_aligned_title():
    fig, ax = plt.subplots()
    try:
        nf.clean_title(ax, "hello")
        assert ax.get_title(loc="left") == "hello"
    finally:
        plt.close(fig)


def test_despine_default_hides_top_right():
    fig, ax = plt.subplots()
    try:
        # Force a known starting state regardless of ambient rcParams/theme.
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
        nf.despine(ax)
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["left"].get_visible() is True
        assert ax.spines["bottom"].get_visible() is True
    finally:
        plt.close(fig)


def test_despine_custom_sides():
    fig, ax = plt.subplots()
    try:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
        nf.despine(ax, top=False, right=False, left=True, bottom=True)
        assert ax.spines["top"].get_visible() is True
        assert ax.spines["right"].get_visible() is True
        assert ax.spines["left"].get_visible() is False
        assert ax.spines["bottom"].get_visible() is False
    finally:
        plt.close(fig)


def test_grid_with_projection():
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs

    fig, axes = nf.grid(1, 2, width=6.0, panel_aspect=0.6, projection=ccrs.Robinson())
    try:
        assert axes.shape == (1, 2)
    finally:
        plt.close(fig)
