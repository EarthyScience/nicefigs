"""`lines`: per-series x via a mapping."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf


def test_lines_per_series_x_mapping():
    ax = nf.lines({"a": [1, 2, 3], "b": [10, 20]}, {"a": [0, 1, 2], "b": [5, 6]})
    try:
        xs = {ln.get_label(): np.asarray(ln.get_xdata()) for ln in ax.lines}
        assert xs["a"].tolist() == [1, 2, 3] and xs["b"].tolist() == [10, 20]
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)
