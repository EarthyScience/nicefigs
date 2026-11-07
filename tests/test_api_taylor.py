"""`taylor`: each model sits at angle arccos(r) and radius std / ref_std."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf


@pytest.mark.parametrize("normalize, expected_r", [(True, 1.0), (False, 2.0)])
def test_taylor_places_model_at_corr_angle_and_std_radius(normalize, expected_r):
    ax = nf.taylor({"m": (2.0, 0.8)}, ref_std=2.0, normalize=normalize, legend=False)
    try:
        model = [ln for ln in ax.lines if ln.get_label() == "m"][0]
        theta, r = model.get_xdata()[0], model.get_ydata()[0]
        assert theta == pytest.approx(np.arccos(0.8))
        assert r == pytest.approx(expected_r)
    finally:
        plt.close(ax.figure)


def test_taylor_requires_correlation():
    with pytest.raises(KeyError):
        nf.taylor([{"name": "m", "std": 1.0}])
