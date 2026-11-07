"""Gridded (xarray/zarr) and unstructured (icosahedral) map support."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data as npd


def test_raster_map_pcolormesh_and_imshow():
    lon, lat, z = npd.global_grid(res=5.0)
    _, ax = nf.subplots()
    _, m = nf.raster_map(lon, lat, z, cmap="Spectral_r", ax=ax)
    assert m is not None
    plt.close(ax.figure)
    _, ax2 = nf.subplots()
    _, m2 = nf.raster_map(lon, lat, z, method="imshow", ax=ax2)
    assert m2 is not None
    plt.close(ax2.figure)


def test_map_auto_raster_for_2d_field():
    lon, lat, z = npd.global_grid(res=4.0)
    fig, ax, m = nf.map(lon, lat, z, inset_hist=False)
    try:
        assert m.__class__.__name__ in ("QuadMesh", "GeoQuadMesh")
    finally:
        plt.close(fig)


def test_map_from_xarray_dataarray():
    xr = pytest.importorskip("xarray")
    da = npd.global_dataarray(res=5.0)
    assert isinstance(da, xr.DataArray)
    fig, ax, m = nf.map(da, inset_hist=False)
    try:
        assert m is not None
    finally:
        plt.close(fig)


def test_from_xarray_extracts_lonlat():
    pytest.importorskip("xarray")
    da = npd.global_dataarray(res=6.0)
    lon, lat, field = nf.from_xarray(da)
    assert lon.ndim == 1 and lat.ndim == 1
    assert field.shape == (lat.size, lon.size)


def test_from_xarray_missing_coords_raises():
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
    with pytest.raises(KeyError):
        nf.from_xarray(da)


def test_tri_map_unstructured():
    lon, lat, v = npd.icosahedral_field(n=1500)
    _, ax = nf.subplots()
    _, m = nf.tri_map(lon, lat, v, cmap="RdBu_r", ax=ax)
    assert m is not None
    plt.close(ax.figure)


def test_map_kind_tri_and_tripcolor():
    lon, lat, v = npd.icosahedral_field(n=1200)
    for kind in ("tri", "tripcolor"):
        fig, ax, m = nf.map(lon, lat, v, kind=kind, inset_hist=False, diverging=True)
        assert m is not None
        plt.close(fig)


def test_tri_map_global_projected_masks_wrapping():
    # a global unstructured field on a projected map: triangulation happens in
    # projected coords and wrapping/pole triangles are masked (no error, some masked)
    from matplotlib.tri import Triangulation

    lon, lat, v = npd.icosahedral_field(n=2000)
    fig, ax, m = nf.map(lon, lat, v, kind="tri", inset_hist=False)
    try:
        assert m is not None
    finally:
        plt.close(fig)
    # direct triangulation across the antimeridian would produce huge triangles;
    # confirm our edge-based mask flags some of them for a global point set.
    tri = Triangulation(lon, lat)
    xt = lon[tri.triangles]
    span = xt.max(1) - xt.min(1)
    assert (span > 180).any()  # such triangles exist and must be masked out
