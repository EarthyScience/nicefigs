"""`caterpillar`: threshold annotations in both orientations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import nicefigs as nf


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_caterpillar_annotates_items_over_threshold(orientation):
    ax = nf.caterpillar(
        ["a", "b", "c"], [0.1, -2.0, 3.0], err=[0.1] * 3, orientation=orientation, annotate_threshold=1.0
    )
    try:
        assert sorted(t.get_text() for t in ax.texts) == ["b", "c"]
    finally:
        plt.close(ax.figure)
