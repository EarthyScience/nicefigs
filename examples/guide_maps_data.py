"""Guide figure: maps from gridded (xarray/zarr) and unstructured data.

Not part of the numbered gallery — referenced from the Maps guide page.
Left: a regular lon/lat field via pcolormesh. Right: an icosahedral-like
unstructured field via tricontourf.
"""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    fig = nf.figure(width="full", aspect=0.3)

    # (a) gridded field (the xarray/zarr case) — pcolormesh
    ax1, ccrs = nf.make_map_axes(fig, (1, 2, 1), projection="robinson")
    lon, lat, z = data.global_grid(res=2.0)
    nf.draw_background(ax1, ccrs)
    ax1.set_extent(nf.maps.ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
    _, m1 = nf.raster_map(lon, lat, z, ccrs=ccrs, cmap="Spectral_r", ax=ax1)
    nf.colorbar(m1, ax1, label="T [°C]")
    nf.panel_label(ax1, "a", "gridded (pcolormesh)")

    # (b) unstructured / icosahedral — tricontourf
    ax2, ccrs = nf.make_map_axes(fig, (1, 2, 2), projection="robinson")
    lo, la, v = data.icosahedral_field()
    lim = nf.symmetric_limits(v)
    nf.draw_background(ax2, ccrs)
    ax2.set_extent(nf.maps.ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
    _, m2 = nf.tri_map(lo, la, v, ccrs=ccrs, cmap="RdBu_r", vmin=lim[0], vmax=lim[1], ax=ax2)
    nf.colorbar(m2, ax2, label="anomaly")
    nf.panel_label(ax2, "b", "unstructured (tricontourf)")
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "guide_maps_data", formats=("svg", "png"))
