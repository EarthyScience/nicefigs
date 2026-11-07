"""Bivariate (2D) colormap — encode two variables at once, with a square legend.

A global scatter coloured by a 2D palette (one variable → x-axis of the colour
square, the other → y-axis), with the key drawn as an inset.
"""

import nicefigs as nf
from nicefigs import data
from nicefigs.colormaps2d import normalize01


def make():
    nf.set_theme("paper")
    lon, lat, temp = data.global_points(field="temperature")
    _, _, precip = data.global_points(field="sparse")
    n = min(len(temp), len(precip))
    lon, lat, temp, precip = lon[:n], lat[:n], temp[:n], precip[:n]

    tx, trange = normalize01(temp)
    py, prange = normalize01(precip)
    colors = nf.bivariate_cmap(tx, py)

    # Build the map ourselves so the points carry the 2D colours directly.
    fig = nf.figure(width="full", aspect=0.52)
    ax, ccrs = nf.make_map_axes(fig, 111, projection="robinson")
    nf.draw_background(ax, ccrs)
    ax.set_extent(nf.maps.ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
    ax.scatter(lon, lat, c=colors, s=9, linewidths=0, transform=ccrs.PlateCarree(), rasterized=True)
    nf.clean_title(ax, "Bivariate encoding")

    iax = ax.inset_axes([0.08, 0.14, 0.15, 0.15])
    iax.patch.set_alpha(0.0)
    nf.bivariate_legend(iax, x_label="temp.", y_label="precip.", x_range=trange, y_range=prange)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "06_bivariate", formats=("png", "pdf"))
