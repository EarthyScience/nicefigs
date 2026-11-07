"""Colour-limit contract of the curtain and Hovmöller helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


@pytest.fixture
def field():
    cs = npd.cross_section_field()
    return cs["lats"], cs["levels"], cs["value"]


def test_cross_section_diff_gets_symmetric_limits_and_diverging_cmap(field):
    x, y, z = field
    _, ax = nf.subplots()
    try:
        _, m = nf.cross_section(x, y, z - z.mean(), diff=True, ax=ax)
        lo, hi = m.get_clim()
        assert lo == pytest.approx(-hi)
        assert m.get_cmap().name.startswith("RdBu_r")
    finally:
        plt.close(ax.figure)


def test_cross_section_explicit_limits_win_and_invert_y_is_optional(field):
    x, y, z = field
    _, ax = nf.subplots()
    try:
        _, m = nf.cross_section(x, y, z, vmin=400.0, vmax=420.0, invert_y=False, ax=ax)
        assert m.get_clim() == (400.0, 420.0)
        assert not ax.yaxis_inverted()
    finally:
        plt.close(ax.figure)


def test_hovmoller_non_robust_limits_span_the_data():
    h = npd.hovmoller_field(n_time=20, n_lat=8)
    ax, m = nf.hovmoller(h["times"], h["lats"], h["value"], robust=False, colorbar=False)
    try:
        assert m.get_clim() == pytest.approx((np.min(h["value"]), np.max(h["value"])))
    finally:
        plt.close(ax.figure)
