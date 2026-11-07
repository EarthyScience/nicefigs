"""Tests for the new plot types: Taylor diagram, Hovmöller, vertical profile,
and the row/column grid-label helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


# ------------------------------------------------------------------- taylor
def test_taylor_runs_and_places_markers():
    stats = npd.taylor_stats()
    ax = nf.taylor(stats, title="skill")
    try:
        assert ax.name == "polar"
        # reference star + one marker per model, each a Line2D
        assert len(ax.lines) >= len(stats) + 1
    finally:
        plt.close(ax.figure)


def test_taylor_accepts_absolute_std_and_dict_input():
    ax = nf.taylor(
        {"A": (0.8, 0.9), "B": (1.1, 0.95)},
        ref_std=1.0,
        normalize=True,
        legend=False,
    )
    try:
        assert ax.name == "polar"
    finally:
        plt.close(ax.figure)


def test_taylor_missing_keys_raise():
    with pytest.raises(KeyError):
        nf.taylor([{"name": "x", "std": 1.0}])  # no correlation


def test_taylor_on_existing_polar_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    out = nf.taylor([{"name": "m", "correlation": 0.9, "std": 1.0}], ax=ax)
    try:
        assert out is ax
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- hovmoller
def test_hovmoller_single_returns_ax_and_mesh():
    h = npd.hovmoller_field()
    ax, m = nf.hovmoller(h["times"], h["lats"], h["value"], colorbar_label="CO2")
    try:
        from matplotlib.collections import QuadMesh

        assert isinstance(m, QuadMesh)
        # datetime x-axis is numeric under the hood
        assert ax.get_xlim()[1] > ax.get_xlim()[0]
    finally:
        plt.close(ax.figure)


def test_hovmoller_numeric_x_no_colorbar_on_supplied_ax():
    fig, ax = plt.subplots()
    z = np.random.default_rng(0).normal(0, 1, (5, 8))
    out_ax, m = nf.hovmoller(np.arange(5), np.linspace(-90, 90, 8), z, ax=ax)
    try:
        assert out_ax is ax
    finally:
        plt.close(fig)


def test_hovmoller_panels_split_colorbars():
    h = npd.hovmoller_field()
    diff = h["value"] - h["reference"]
    fig, axes = nf.hovmoller_panels(
        {"Model": h["value"], "Ref": h["reference"], "Diff": diff},
        h["times"],
        h["lats"],
        diff_fields={"Diff"},
        value_label="CO2",
        diff_label="Δ",
    )
    try:
        assert len(axes) == 3
    finally:
        plt.close(fig)


def test_hovmoller_panels_default_titles_from_dict_keys():
    """Default behavior (titles=None): titles come from dict keys."""
    h = npd.hovmoller_field()
    diff = h["value"] - h["reference"]
    fig, axes = nf.hovmoller_panels(
        {"Model": h["value"], "Ref": h["reference"], "Diff": diff},
        h["times"],
        h["lats"],
        diff_fields={"Diff"},
    )
    try:
        # Default: panel titles are the field names from dict keys, set with loc="left"
        assert axes[0].get_title(loc="left") == "Model"
        assert axes[1].get_title(loc="left") == "Ref"
        assert axes[2].get_title(loc="left") == "Diff"
    finally:
        plt.close(fig)


def test_hovmoller_panels_custom_titles_override_dict_keys():
    """With titles parameter: custom titles override dict keys; use label_panels for letters."""
    h = npd.hovmoller_field()
    diff = h["value"] - h["reference"]
    fig, axes = nf.hovmoller_panels(
        {"Model": h["value"], "Ref": h["reference"], "Diff": diff},
        h["times"],
        h["lats"],
        diff_fields={"Diff"},
        titles=["Model Output", "Reference Data", "Model - Reference"],
    )
    try:
        # With titles parameter: use the provided titles, set with loc="left"
        assert axes[0].get_title(loc="left") == "Model Output"
        assert axes[1].get_title(loc="left") == "Reference Data"
        assert axes[2].get_title(loc="left") == "Model - Reference"
        # Caller can then label with letters via label_panels
        nf.label_panels(axes)
        # After label_panels, each axes should have letter labels
        assert len(axes[0].texts) >= 1  # label text is present
        assert len(axes[1].texts) >= 1
        assert len(axes[2].texts) >= 1
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ profile
def test_profile_inverts_y_and_plots_series():
    vp = npd.vertical_profiles()
    ax = nf.profile(
        vp["profiles"],
        vp["levels"],
        baselines={"CT": vp["baseline"]},
        xlabel="CO2",
        xunits="ppmv",
        ylabel="pressure",
        yunits="hPa",
    )
    try:
        assert ax.yaxis_inverted()
        assert len(ax.lines) == len(vp["profiles"]) + 1  # + baseline
    finally:
        plt.close(ax.figure)


def test_profile_single_array_and_log():
    ax = nf.profile(np.linspace(400, 410, 12), log_y=True)
    try:
        assert ax.get_yscale() == "log"
    finally:
        plt.close(ax.figure)


# --------------------------------------------------------- row/col labels
def test_row_and_col_labels_on_grid():
    fig, axes = nf.grid(2, 3, width="double")
    nf.row_labels(axes, ["r0", "r1"])
    nf.col_labels(axes, ["c0", "c1", "c2"])
    try:
        row_texts = {t.get_text() for t in fig.texts}
        assert {"r0", "r1", "c0", "c1", "c2"}.issubset(
            row_texts | {t.get_text() for ax in axes.ravel() for t in ax.texts}
        )
    finally:
        plt.close(fig)
