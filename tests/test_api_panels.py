"""Contracts of the small-multiples, legend-row and reference-line helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf


def test_facet_calls_draw_once_per_item_in_order_and_hides_unused():
    seen = []
    fig, axes = nf.facet(["a", "b", "c", "d"], lambda ax, item, i: seen.append((item, i)), ncols=3)
    try:
        assert seen == [("a", 0), ("b", 1), ("c", 2), ("d", 3)]
        assert axes.shape == (6,)
        assert [ax.axison for ax in axes] == [True] * 4 + [False] * 2
    finally:
        plt.close(fig)


def test_facet_titles_and_labels_are_drawn():
    fig, axes = nf.facet(range(2), lambda ax, item, i: None, titles=["first", "second"], labels=True)
    try:
        assert axes[0].get_title(loc="left") == "first"
        texts = {t.get_text() for t in axes[1].texts}
        assert "b" in texts
    finally:
        plt.close(fig)


def test_facet_caps_columns_at_item_count():
    fig, axes = nf.facet(range(2), lambda ax, item, i: None, ncols=5)
    try:
        assert len(axes) == 2
    finally:
        plt.close(fig)


def test_grid_with_legend_row_returns_hidden_full_width_axes():
    fig, axes, ax_leg = nf.grid(2, 3, with_legend_row=True)
    try:
        assert axes.shape == (2, 3)
        assert not ax_leg.axison
        fig.canvas.draw()
        leg, left, right = ax_leg.get_position(), axes[1, 0].get_position(), axes[1, 2].get_position()
        assert leg.y1 < left.y0
        assert leg.x0 == pytest.approx(left.x0) and leg.x1 == pytest.approx(right.x1)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_reference_line_draws_line_at_value_with_formatted_label(axis):
    fig, ax = nf.subplots()
    try:
        nf.reference_line(ax, 0.25, axis=axis, label="chance", fmt="{label} ({value:.2f})")
        (line,) = ax.lines
        coords = line.get_xdata() if axis == "x" else line.get_ydata()
        assert np.allclose(coords, 0.25)
        assert [t.get_text().strip() for t in ax.texts] == ["chance (0.25)"]
    finally:
        plt.close(fig)


def test_reference_line_without_label_draws_no_text():
    fig, ax = nf.subplots()
    try:
        nf.reference_line(ax, 1.0)
        assert len(ax.lines) == 1 and not ax.texts
    finally:
        plt.close(fig)
