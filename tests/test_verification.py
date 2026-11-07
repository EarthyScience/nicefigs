"""Tests for the generative-model / ensemble-forecast verification primitives:
reliability diagrams, rank histograms, and precision-recall curves — plus the
multi-band extension to :func:`nicefigs.timeseries`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf


# ---------------------------------------------------------------- reliability
def test_reliability_single_curve_perfect_calibration():
    nominal = np.linspace(0, 1, 11)
    ax = nf.reliability(nominal, nominal.copy(), highlight=None)
    try:
        # diagonal + the one named curve → 2 lines
        assert len(ax.lines) == 2
        # perfectly calibrated curve lies exactly on the y = x diagonal
        _diag, curve = ax.lines
        np.testing.assert_allclose(curve.get_ydata(), curve.get_xdata())
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)
    finally:
        plt.close(ax.figure)


def test_reliability_multiple_named_curves_and_legend():
    nominal = np.linspace(0, 1, 6)
    observed = {
        "underdispersed": np.clip(nominal * 1.2, 0, 1),
        "overdispersed": np.clip(nominal * 0.8, 0, 1),
    }
    ax = nf.reliability(nominal, observed, highlight=0.90)
    try:
        # diagonal + highlight marker + 2 named curves
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "underdispersed" in labels and "overdispersed" in labels
        # diagonal + highlight marker + highlight axvline + 2 named curves
        assert len(ax.lines) == 3 + 2
    finally:
        plt.close(ax.figure)


def test_reliability_serves_pit_calibration_use_case():
    # PIT calibration: empirical CDF of PIT values evaluated at probability levels.
    rng = np.random.default_rng(0)
    pit = rng.uniform(0, 1, 2000)  # well-calibrated ensemble → PIT ~ Uniform(0,1)
    levels = np.linspace(0, 1, 11)
    ecdf = np.array([(pit <= lv).mean() for lv in levels])
    ax = nf.reliability(levels, {"model": ecdf})
    try:
        curve = ax.lines[-1]
        np.testing.assert_allclose(curve.get_ydata(), ecdf)
    finally:
        plt.close(ax.figure)


# ----------------------------------------------------------------- rank hist
def test_rank_histogram_counts_and_normalization():
    ranks = np.array([0, 0, 1, 1, 1, 2])  # n_ranks inferred = 3
    ax = nf.rank_histogram(ranks, normalize=False)
    try:
        heights = sorted(p.get_height() for p in ax.patches)
        assert heights == [1.0, 2.0, 3.0]
    finally:
        plt.close(ax.figure)

    ax2 = nf.rank_histogram(ranks, normalize=True)
    try:
        heights = sorted(p.get_height() for p in ax2.patches)
        np.testing.assert_allclose(heights, [1 / 6, 2 / 6, 3 / 6])
    finally:
        plt.close(ax2.figure)


def test_rank_histogram_ideal_reference_line():
    ranks = np.array([0, 1, 2, 3])
    ax = nf.rank_histogram(ranks, n_ranks=4, ideal=True)
    try:
        assert len(ax.lines) == 1  # the flat-ideal reference line
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.25, 0.25])
    finally:
        plt.close(ax.figure)

    ax2 = nf.rank_histogram(ranks, n_ranks=4, ideal=False)
    try:
        assert len(ax2.lines) == 0
    finally:
        plt.close(ax2.figure)


def test_rank_histogram_summary_annotation():
    ax = nf.rank_histogram(np.array([0, 1, 1, 2]), summary=("delta", 0.12))
    try:
        texts = [t.get_text() for t in ax.texts]
        assert any("0.12" in t for t in texts)
    finally:
        plt.close(ax.figure)


def test_rank_histogram_panels_shared_bins():
    rng = np.random.default_rng(1)
    ranks_by_name = {
        "model A": rng.integers(0, 5, size=200),
        "model B": rng.integers(0, 5, size=200),
    }
    fig, axes = nf.rank_histogram_panels(ranks_by_name, ncols=2)
    try:
        assert len(axes) == 2
        # same shared n_ranks (5) → same x-limits on both panels
        assert axes[0].get_xlim() == axes[1].get_xlim()
    finally:
        plt.close(fig)


def test_rank_histogram_panels_default_titles_from_dict_keys():
    """Default behavior (titles=None): titles come from dict keys."""
    rng = np.random.default_rng(1)
    ranks_by_name = {
        "model A": rng.integers(0, 5, size=200),
        "model B": rng.integers(0, 5, size=200),
    }
    fig, axes = nf.rank_histogram_panels(ranks_by_name, ncols=2)
    try:
        # Default: panel titles are the model names from dict keys, set with loc="left"
        assert axes[0].get_title(loc="left") == "model A"
        assert axes[1].get_title(loc="left") == "model B"
    finally:
        plt.close(fig)


def test_rank_histogram_panels_custom_titles_override_dict_keys():
    """With titles parameter: custom titles override dict keys; use label_panels for letters."""
    rng = np.random.default_rng(1)
    ranks_by_name = {
        "model A": rng.integers(0, 5, size=200),
        "model B": rng.integers(0, 5, size=200),
    }
    fig, axes = nf.rank_histogram_panels(ranks_by_name, ncols=2, titles=["Custom A", "Custom B"])
    try:
        # With titles parameter: use the provided titles, set with loc="left"
        assert axes[0].get_title(loc="left") == "Custom A"
        assert axes[1].get_title(loc="left") == "Custom B"
        # Caller can then label with letters via label_panels
        nf.label_panels(axes)
        # After label_panels, each axes should have letter labels
        assert len(axes[0].texts) >= 1  # label text is present
        assert len(axes[1].texts) >= 1
    finally:
        plt.close(fig)


# ------------------------------------------------------------ precision-recall
def test_precision_recall_parametric_curves():
    recall = np.linspace(0, 1, 20)
    precision_a = 1 - 0.3 * recall
    precision_b = 1 - 0.6 * recall
    ax = nf.precision_recall({"A": (recall, precision_a), "B": (recall, precision_b)}, iso_f1=None)
    try:
        # 2 named model lines only (no iso-F1 curves this time)
        curve_lines = [ln for ln in ax.lines if ln.get_label() in ("A", "B")]
        assert len(curve_lines) == 2
        np.testing.assert_allclose(curve_lines[0].get_ydata(), precision_a)
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)
    finally:
        plt.close(ax.figure)


def test_precision_recall_single_operating_points():
    ax = nf.precision_recall({"A": (0.7, 0.8), "B": (0.5, 0.9)}, iso_f1=None)
    try:
        pts = [ln for ln in ax.lines if ln.get_linestyle() == "None" or ln.get_linestyle() == "none"]
        assert len(pts) == 2
        xy = sorted((ln.get_xdata()[0], ln.get_ydata()[0]) for ln in pts)
        assert xy == [(0.5, 0.9), (0.7, 0.8)]
    finally:
        plt.close(ax.figure)


def test_precision_recall_iso_f1_and_diagonal():
    ax = nf.precision_recall({"A": (0.6, 0.6)}, iso_f1=(0.5,), diagonal=0.3)
    try:
        # iso-F1 dotted curve + the no-skill diagonal + the model point
        assert len(ax.lines) >= 3
        labels = [ln.get_label() for ln in ax.lines]
        assert "no skill" in labels
    finally:
        plt.close(ax.figure)


# ------------------------------------------------- ensemble spread bands (timeseries)
def test_timeseries_single_band_backward_compatible():
    import datetime as dt

    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    y = np.linspace(0, 1, 10)
    ax = nf.timeseries(dates, {"ens": y}, bands={"ens": (y - 0.1, y + 0.1)})
    try:
        assert len(ax.collections) == 1
    finally:
        plt.close(ax.figure)


def test_timeseries_multiple_quantile_bands_layered_by_width():
    import datetime as dt

    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    y = np.zeros(10)
    narrow = (y - 0.5, y + 0.5)
    wide = (y - 2.0, y + 2.0)
    ax = nf.timeseries(dates, {"ens": y}, bands={"ens": [narrow, wide]})
    try:
        # two fill_between polygons drawn for the two nested bands
        assert len(ax.collections) == 2
        alphas = [c.get_alpha() for c in ax.collections]
        # the wider band is drawn first (further back / lower alpha), narrower on top
        assert alphas[0] < alphas[1]
    finally:
        plt.close(ax.figure)


def test_timeseries_multiple_ensembles_with_bands_and_truth_line():
    import datetime as dt

    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    y1 = np.linspace(0, 1, 10)
    y2 = np.linspace(1, 0, 10)
    truth = (y1 + y2) / 2
    series = {"ens1": y1, "ens2": y2, "truth": truth}
    bands = {"ens1": (y1 - 0.1, y1 + 0.1), "ens2": (y2 - 0.1, y2 + 0.1)}
    ax = nf.timeseries(dates, series, bands=bands, highlight="truth")
    try:
        assert len(ax.lines) == 3
        assert len(ax.collections) == 2
    finally:
        plt.close(ax.figure)
