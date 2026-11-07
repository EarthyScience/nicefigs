"""Shared pytest fixtures/config for the nicefigs test suite.

Forces a non-interactive backend (Agg) before anything imports pyplot, and
makes sure every test starts and ends with no stray open figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Reset matplotlib rcParams/theme state and close all figures after each test."""
    import nicefigs as nf

    yield
    plt.close("all")
    # Reset the global theme so tests don't leak style state into each other.
    nf.set_theme("paper")
