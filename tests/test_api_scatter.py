"""`scatter`: a fixed marker size passed as ``s`` applies to every group."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf


def test_scatter_s_kwarg_applies_to_all_groups():
    _, ax = nf.subplots()
    try:
        _, colls = nf.scatter([0, 1, 2, 3], [0, 1, 2, 3], group=["a", "b", "c", "a"], s=7, ax=ax)
        assert {float(np.unique(c.get_sizes())[0]) for c in colls.values()} == {7.0}
    finally:
        plt.close(ax.figure)
