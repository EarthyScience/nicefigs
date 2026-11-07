"""comparison.py: model_comparison violin/box grids."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


def _small_scores():
    # model_scores has a hardcoded per-model "skill" table, so custom model
    # names must be drawn from {"Persistence", "Climatology", "U-Net", "Transformer"}.
    return npd.model_scores(
        seed=0,
        models=["Persistence", "U-Net"],
        metrics=["RMSE", "Bias"],
        timescales=["daily", "monthly"],
        n=20,
    )


@pytest.mark.parametrize("kind", ["violin", "box", "strip"])
def test_model_comparison_kinds_smoke(kind):
    meta = _small_scores()
    fig, axes = nf.model_comparison(meta, kind=kind)
    try:
        assert len(axes) == len(meta["metrics"])
    finally:
        plt.close(fig)


def test_model_comparison_accepts_plain_scores_dict():
    meta = _small_scores()
    fig, axes = nf.model_comparison(
        meta["scores"], models=meta["models"], metrics=meta["metrics"], groups=meta["timescales"]
    )
    try:
        assert len(axes) == 2
    finally:
        plt.close(fig)


def test_model_comparison_without_groups():
    rng = np.random.default_rng(0)
    scores = {
        "RMSE": {"A": rng.normal(1, 0.1, 30), "B": rng.normal(1.5, 0.1, 30)},
        "Bias": {"A": rng.normal(0, 0.1, 30), "B": rng.normal(0.2, 0.1, 30)},
    }
    fig, axes = nf.model_comparison(scores)
    try:
        assert len(axes) == 2
        # no legend expected without groups
    finally:
        plt.close(fig)


def test_model_comparison_metric_lims_and_units():
    meta = _small_scores()
    fig, axes = nf.model_comparison(
        meta, metric_units={"RMSE": "m", "Bias": "m"}, metric_lims={"RMSE": (0, 5)}
    )
    try:
        ax0 = axes[0]
        assert ax0.get_ylim() == (0, 5) or ax0.get_ylim() == pytest.approx((0, 5))
    finally:
        plt.close(fig)


def test_model_comparison_extra_axes_turned_off():
    meta = npd.model_scores(seed=0, models=["Persistence", "U-Net"], metrics=["M1", "M2", "M3"], n=15)
    fig, axes = nf.model_comparison(meta, ncols=2)
    try:
        assert len(axes) == 4  # 3 metrics -> 2x2 grid, 4th axis off
        assert axes[3].axison is False
    finally:
        plt.close(fig)
