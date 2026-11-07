"""Contracts of `grouped_bars` / `stacked_bars`: bar heights and stacking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf


@pytest.fixture
def ax():
    _, ax = nf.subplots()
    yield ax
    plt.close(ax.figure)


def test_grouped_dict_series_zero_fills_missing_categories(ax):
    nf.grouped_bars(["a", "b", "c"], {"s1": {"a": 1.0, "c": 3.0}, "s2": [4.0, 5.0, 6.0]}, ax=ax)
    heights = [p.get_height() for p in ax.patches]
    assert heights == [1.0, 0.0, 3.0, 4.0, 5.0, 6.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_stacked_tops_equal_category_totals(ax):
    segs = {"x": [1.0, 2.0], "y": [3.0, 0.5]}
    nf.stacked_bars(["p", "q"], segs, ax=ax)
    tops = {}
    for p in ax.patches:
        cx = round(p.get_x() + p.get_width() / 2, 6)
        tops[cx] = max(tops.get(cx, 0.0), p.get_y() + p.get_height())
    assert sorted(tops.values()) == pytest.approx([2.5, 4.0])


def test_stacked_normalize_keeps_empty_category_at_zero(ax):
    nf.stacked_bars(["p", "q"], {"x": [1.0, 0.0], "y": [3.0, 0.0]}, normalize=True, ax=ax)
    tops = [p.get_y() + p.get_height() for p in ax.patches]
    assert max(tops) == pytest.approx(1.0)
    assert np.isfinite(tops).all()
