"""timeseries.py: date-axis line plots."""

from __future__ import annotations

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


def test_timeseries_smoke_multi_series():
    dates, series = npd.timeseries(n_days=100, series=["obs", "modelA"])
    ax = nf.timeseries(dates, series, ylabel="Temperature", yunits="degC", title="ts")
    try:
        assert len(ax.lines) == 2
        assert ax.get_ylabel() == "Temperature [°C]"
    finally:
        plt.close(ax.figure)


def test_timeseries_single_array_becomes_one_series():
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    vals = np.linspace(0, 1, 30)
    ax = nf.timeseries(dates, vals)
    try:
        assert len(ax.lines) == 1
    finally:
        plt.close(ax.figure)


def test_timeseries_no_legend_for_single_series():
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    vals = np.arange(10.0)
    ax = nf.timeseries(dates, {"only": vals})
    try:
        assert ax.get_legend() is None
    finally:
        plt.close(ax.figure)


def test_timeseries_legend_present_for_multi_series():
    dates, series = npd.timeseries(n_days=60, series=["a", "b", "c"])
    ax = nf.timeseries(dates, series)
    try:
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)


def test_timeseries_bands_add_fill_between():
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(20)]
    vals = np.arange(20.0)
    lo = vals - 1
    hi = vals + 1
    ax = nf.timeseries(dates, {"s": vals}, bands={"s": (lo, hi)})
    try:
        assert len(ax.collections) >= 1
    finally:
        plt.close(ax.figure)


def test_timeseries_highlight_thicker_line():
    dates, series = npd.timeseries(n_days=60, series=["obs", "model"])
    ax = nf.timeseries(dates, series, highlight="obs")
    try:
        widths = [ln.get_linewidth() for ln in ax.lines]
        assert max(widths) == pytest.approx(widths[[ln.get_label() for ln in ax.lines].index("obs")])
    finally:
        plt.close(ax.figure)


def test_timeseries_on_existing_axes():
    fig, ax = plt.subplots()
    try:
        dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
        out = nf.timeseries(dates, {"s": np.arange(10.0)}, ax=ax)
        assert out is ax
    finally:
        plt.close(fig)


@pytest.mark.parametrize("n_days", [10, 40, 200, 1500])
def test_timeseries_date_axis_formatting_for_various_spans(n_days):
    dates, series = npd.timeseries(n_days=n_days, series=["obs"])
    ax = nf.timeseries(dates, series)
    try:
        assert ax.xaxis.get_major_formatter() is not None
    finally:
        plt.close(ax.figure)


def test_timeseries_obs_hue_colours_markers_by_a_third_variable():
    dates, series = npd.timeseries(n_days=60, series=["model"])
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(60, 20, replace=False))
    odates = [dates[i] for i in idx]
    ovals = series["model"][idx]
    hue = rng.uniform(0, 12000, 20)
    ax = nf.timeseries(dates, series, obs=(odates, ovals), obs_hue=hue, obs_cmap="viridis")
    try:
        # one PathCollection for the hue-mapped obs, carrying per-point colours
        colls = [c for c in ax.collections if c.get_array() is not None]
        assert len(colls) == 1
        assert len(colls[0].get_array()) == 20
    finally:
        plt.close(ax.figure)


def test_timeseries_obs_hue_legend_gets_one_proxy_handle_not_per_point():
    dates, series = npd.timeseries(n_days=60, series=["model"])
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(60, 15, replace=False))
    odates = [dates[i] for i in idx]
    ovals = series["model"][idx]
    hue = rng.uniform(0, 1, 15)
    ax = nf.timeseries(dates, series, obs=(odates, ovals), obs_hue=hue, obs_label="obs")
    try:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels.count("obs") == 1
        assert "model" in labels
    finally:
        plt.close(ax.figure)


def test_timeseries_obs_hue_draws_colorbar_when_requested():
    dates, series = npd.timeseries(n_days=60, series=["model"])
    rng = np.random.default_rng(2)
    idx = np.sort(rng.choice(60, 15, replace=False))
    odates = [dates[i] for i in idx]
    ovals = series["model"][idx]
    hue = rng.uniform(0, 1, 15)
    ax_no_cbar = nf.timeseries(dates, series, obs=(odates, ovals), obs_hue=hue)
    n_axes_before = len(ax_no_cbar.figure.axes)
    plt.close(ax_no_cbar.figure)
    ax = nf.timeseries(dates, series, obs=(odates, ovals), obs_hue=hue, obs_colorbar=True)
    try:
        assert len(ax.figure.axes) > n_axes_before
    finally:
        plt.close(ax.figure)


def test_timeseries_numpy_datetime64_axis_gets_date_formatting():
    dates = np.array("2020-01-01", dtype="datetime64[D]") + np.arange(400)
    vals = np.linspace(0, 1, 400)
    ax = nf.timeseries(dates, {"s": vals})
    try:
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    finally:
        plt.close(ax.figure)


def test_timeseries_without_obs_hue_still_uses_flat_obs_color():
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    vals = np.arange(10.0)
    ax = nf.timeseries(dates, {"s": vals}, obs=(dates, vals), obs_color="#ff0000")
    try:
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)
