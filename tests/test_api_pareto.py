"""Contract of `frontier`: exactly the non-dominated points, under either preference."""

from __future__ import annotations

import numpy as np
import pytest

import nicefigs as nf


def _brute_force(x, y, minimize_x, maximize_y):
    sx = 1 if minimize_x else -1
    sy = -1 if maximize_y else 1
    a, b = sx * x, sy * y  # both "lower is better" now
    keep = []
    for i in range(x.size):
        dominated = (a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))
        if not dominated.any():
            keep.append(i)
    return set(keep)


@pytest.mark.parametrize("minimize_x", [True, False])
@pytest.mark.parametrize("maximize_y", [True, False])
def test_frontier_matches_brute_force(minimize_x, maximize_y):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=60), rng.normal(size=60)
    idx = nf.frontier(
        x,
        y,
        xdirection="lower_better" if minimize_x else "higher_better",
        ydirection="higher_better" if maximize_y else "lower_better",
    )
    assert set(idx.tolist()) == _brute_force(x, y, minimize_x, maximize_y)
    assert np.all(np.diff(x[idx]) >= 0)


def test_frontier_keeps_tied_duplicates():
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([1.0, 1.0, 0.0])
    assert sorted(nf.frontier(x, y).tolist()) == [0, 1]


def test_baseline_labels_sit_at_the_left_edge():
    import matplotlib.pyplot as plt

    ax, _ = nf.pareto([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], baselines={"ref": 2.5})
    try:
        labels = [t for t in ax.texts if t.get_text().startswith("ref")]
        assert len(labels) == 1 and labels[0].get_ha() == "left"
    finally:
        plt.close(ax.figure)
