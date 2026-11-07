"""kde.py: agreement (predicted-vs-observed) plot and numeric correctness."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


def test_agreement_returns_axes_and_stats_dict():
    x, y = npd.xy_agreement(n=400, seed=0)
    ax, stats = nf.agreement(x, y)
    try:
        assert ax is not None
        for key in ("slope", "intercept", "r2", "rmse", "bias", "n"):
            assert key in stats
        assert stats["n"] <= 400
        assert np.isfinite(stats["r2"])
        assert np.isfinite(stats["rmse"])
        assert np.isfinite(stats["bias"])
    finally:
        plt.close(ax.figure)


def test_agreement_y_equals_x_gives_perfect_fit():
    # Use density="hist" (not the default "kde"): a perfectly-correlated cloud
    # gives gaussian_kde a singular covariance matrix and raises LinAlgError,
    # which is orthogonal to what this test checks (the fit statistics).
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 300)
    y = x.copy()
    ax, stats = nf.agreement(x, y, density="hist")
    try:
        assert stats["r2"] == pytest.approx(1.0, abs=1e-6)
        assert stats["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert stats["bias"] == pytest.approx(0.0, abs=1e-6)
        assert stats["slope"] == pytest.approx(1.0, abs=1e-6)
        assert stats["intercept"] == pytest.approx(0.0, abs=1e-6)
    finally:
        plt.close(ax.figure)


def test_agreement_kde_degrades_on_collinear_cloud():
    # A perfectly collinear cloud (y = x + c) gives gaussian_kde a singular
    # covariance; the default density="kde" must fall back to hexbin instead of
    # raising "Contour levels must be increasing".
    rng = np.random.default_rng(3)
    x = rng.normal(400, 5, 250)
    ax, stats = nf.agreement(x, x + 1.0)  # default density="kde"
    try:
        assert stats["r2"] == pytest.approx(1.0, abs=1e-6)
        assert stats["bias"] == pytest.approx(1.0, abs=1e-6)
    finally:
        plt.close(ax.figure)


def test_agreement_known_bias():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 300)
    y = x + 2.0  # constant offset, no noise
    ax, stats = nf.agreement(x, y, density="hist")
    try:
        assert stats["bias"] == pytest.approx(2.0, abs=1e-6)
        assert stats["rmse"] == pytest.approx(2.0, abs=1e-6)
    finally:
        plt.close(ax.figure)


def test_agreement_drops_nans():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 100)
    y = x + rng.normal(0, 0.3, x.shape)
    x[5] = np.nan
    y[10] = np.nan
    ax, stats = nf.agreement(x, y)
    try:
        assert stats["n"] == 98
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("density", ["kde", "hexbin", "hist"])
def test_agreement_density_modes_smoke(density):
    x, y = npd.xy_agreement(n=200, seed=3)
    ax, stats = nf.agreement(x, y, density=density, gridsize=20)
    try:
        assert stats["n"] == 200
    finally:
        plt.close(ax.figure)


def test_agreement_on_existing_axes():
    fig, ax = plt.subplots()
    try:
        x, y = npd.xy_agreement(n=100, seed=4)
        ax_out, stats = nf.agreement(x, y, ax=ax)
        assert ax_out is ax
    finally:
        plt.close(fig)


def test_agreement_metrics_text_box_present_when_metrics_true():
    x, y = npd.xy_agreement(n=100, seed=5)
    ax, stats = nf.agreement(x, y, metrics=True)
    try:
        assert len(ax.texts) >= 1
    finally:
        plt.close(ax.figure)


def test_agreement_no_metrics_text_when_disabled():
    x, y = npd.xy_agreement(n=100, seed=6)
    ax, stats = nf.agreement(x, y, metrics=False)
    try:
        assert len(ax.texts) == 0
    finally:
        plt.close(ax.figure)
