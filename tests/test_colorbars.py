"""colorbars.py: slim colorbar placement, make_mappable, inset histogram colorbar."""

from __future__ import annotations

import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nicefigs.colorbars import colorbar, inset_histogram_colorbar, make_mappable


def test_colorbar_single_axes_returns_colorbar_and_cax():
    fig, ax = plt.subplots()
    try:
        im = ax.imshow(np.random.default_rng(0).normal(size=(10, 10)))
        n_axes_before = len(fig.axes)
        cb = colorbar(im, ax, label="value")
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
        assert len(fig.axes) == n_axes_before + 1
        assert cb.ax.get_xlabel() or cb.ax.get_ylabel()
    finally:
        plt.close(fig)


def test_colorbar_spans_list_of_axes():
    fig, axes = plt.subplots(1, 3)
    try:
        im = None
        for ax in axes:
            im = ax.imshow(np.random.default_rng(0).normal(size=(5, 5)))
        cb = colorbar(im, list(axes))
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("location", ["right", "left", "top", "bottom"])
def test_colorbar_all_locations(location):
    fig, ax = plt.subplots()
    try:
        im = ax.imshow(np.random.default_rng(0).normal(size=(5, 5)))
        cb = colorbar(im, ax, location=location)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
    finally:
        plt.close(fig)


def test_make_mappable_from_values():
    sm = make_mappable(values=[1.0, 2.0, 3.0, 10.0], cmap="viridis")
    assert sm.norm.vmin == 1.0
    assert sm.norm.vmax == 10.0
    assert sm.cmap.name == "viridis"


def test_make_mappable_explicit_vmin_vmax():
    sm = make_mappable(vmin=-3, vmax=3)
    assert sm.norm.vmin == -3
    assert sm.norm.vmax == 3


def test_make_mappable_can_feed_colorbar():
    fig, ax = plt.subplots()
    try:
        sm = make_mappable(values=[0, 1, 2])
        cb = colorbar(sm, ax)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
    finally:
        plt.close(fig)


def test_inset_histogram_colorbar_returns_inset_axes():
    fig, ax = plt.subplots()
    try:
        vals = np.random.default_rng(0).normal(size=500)
        inset = inset_histogram_colorbar(ax, vals, label="value")
        assert inset is not None
        assert inset in ax.child_axes
        assert inset.get_xlabel() == "value"
    finally:
        plt.close(fig)


def test_inset_histogram_colorbar_empty_input_returns_none():
    fig, ax = plt.subplots()
    try:
        inset = inset_histogram_colorbar(ax, np.full(10, np.nan))
        assert inset is None
        inset2 = inset_histogram_colorbar(ax, np.array([]))
        assert inset2 is None
    finally:
        plt.close(fig)
