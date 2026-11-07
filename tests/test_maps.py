"""maps.py: smoke-render scatter/binned/regional/diverging maps.

Cartopy coastline data may need a network download on first use in some
environments. These tests must still pass offline, since ``draw_background``
swallows coastline failures (``except Exception: pass``) and the package
falls back to a plain equirectangular axes entirely if cartopy itself is
unavailable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import nicefigs as nf
from nicefigs import data as npd


def test_map_scatter_small_n_returns_expected_types():
    lon, lat, vals = npd.global_points(n=300, seed=1)
    fig, ax, mappable = nf.map(lon, lat, vals, kind="scatter")
    try:
        assert fig is not None
        assert ax is not None
        assert mappable is not None
    finally:
        plt.close(fig)


def test_map_auto_kind_uses_scatter_below_threshold():
    lon, lat, vals = npd.global_points(n=200, seed=1)
    fig, ax, mappable = nf.map(lon, lat, vals, kind="auto")
    try:
        import matplotlib.collections as mcoll

        assert isinstance(mappable, mcoll.PathCollection)
    finally:
        plt.close(fig)


def test_map_binned_kind():
    lon, lat, vals = npd.global_points(n=500, seed=2)
    fig, ax, mappable = nf.map(lon, lat, vals, kind="binned")
    try:
        assert mappable is not None
    finally:
        plt.close(fig)


def test_map_regional_extent():
    lon2d, lat2d, z = npd.regional_grid(seed=0, extent=(5, 15, 45, 50), res=0.5)
    fig, ax, mappable = nf.map(
        lon2d.ravel(), lat2d.ravel(), z.ravel(), extent=(5, 15, 45, 50), kind="scatter"
    )
    try:
        assert mappable is not None
    finally:
        plt.close(fig)


def test_map_diverging():
    lon, lat, vals = npd.global_points(n=300, seed=3, field="anomaly")
    fig, ax, mappable = nf.map(lon, lat, vals, diverging=True)
    try:
        import matplotlib.colors as mcolors

        assert isinstance(mappable.norm, mcolors.TwoSlopeNorm)
        assert mappable.norm.vcenter == 0.0
    finally:
        plt.close(fig)


def test_map_no_values_plots_locations_only():
    lon, lat, _ = npd.global_points(n=100, seed=4)
    fig, ax, mappable = nf.map(lon, lat, None)
    try:
        assert mappable is None
    finally:
        plt.close(fig)


def test_map_on_existing_axes():
    fig, ax = plt.subplots()
    try:
        lon, lat, vals = npd.global_points(n=100, seed=5)
        fig2, ax2, mappable = nf.map(lon, lat, vals, ax=ax)
        assert ax2 is ax
        assert fig2 is fig
    finally:
        plt.close(fig)


def test_map_with_conventional_colorbar():
    lon, lat, vals = npd.global_points(n=100, seed=6)
    fig, ax, mappable = nf.map(lon, lat, vals, cbar=True, colorbar_label="value")
    try:
        # a conventional colorbar axes should have been added
        assert len(fig.axes) >= 2
    finally:
        plt.close(fig)


def test_map_with_title():
    lon, lat, vals = npd.global_points(n=50, seed=7)
    fig, ax, mappable = nf.map(lon, lat, vals, title="My map")
    try:
        assert ax.get_title(loc="left") == "My map"
    finally:
        plt.close(fig)


def test_scatter_map_returns_pathcollection():
    from nicefigs.maps import scatter_map

    fig, ax = plt.subplots()
    try:
        lon, lat, vals = npd.global_points(n=50, seed=8)
        _, pc = scatter_map(lon, lat, vals, ax=ax)
        import matplotlib.collections as mcoll

        assert isinstance(pc, mcoll.PathCollection)
    finally:
        plt.close(fig)


def test_binned_map_returns_quadmesh():
    from nicefigs.maps import binned_map

    fig, ax = plt.subplots()
    try:
        lon, lat, vals = npd.global_points(n=500, seed=9)
        _, qm = binned_map(lon, lat, vals, bin_deg=10.0, ax=ax)
        import matplotlib.collections as mcoll

        assert isinstance(qm, mcoll.QuadMesh)
    finally:
        plt.close(fig)


def test_make_map_axes_degrades_gracefully_without_cartopy(monkeypatch):
    """Simulate cartopy being unavailable: make_map_axes must still work."""
    import nicefigs.maps as maps_mod

    monkeypatch.setattr(maps_mod, "_try_cartopy", lambda: None)
    fig = plt.figure()
    try:
        ax, ccrs = maps_mod.make_map_axes(fig)
        assert ccrs is None
        assert ax is not None
    finally:
        plt.close(fig)


def test_draw_background_noop_without_cartopy():
    from nicefigs.maps import draw_background

    fig, ax = plt.subplots()
    try:
        # Should not raise even though ax isn't a GeoAxes, because ccrs=None.
        draw_background(ax, None)
    finally:
        plt.close(fig)


def test_map_offline_safe_even_if_coastline_download_fails(monkeypatch):
    """draw_background swallows coastline exceptions -- verify that directly."""
    pytest.importorskip("cartopy")

    class BoomAxes:
        def add_feature(self, *a, **k):
            pass

        def coastlines(self, *a, **k):
            raise OSError("simulated network failure")

        @property
        def spines(self):
            raise AttributeError

    from nicefigs.maps import draw_background

    class FakeCcrs:
        pass

    # cartopy present (truthy ccrs) but coastlines() raises -- must not propagate.
    draw_background(BoomAxes(), FakeCcrs())
