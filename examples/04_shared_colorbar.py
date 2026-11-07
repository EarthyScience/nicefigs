"""Shared colorbar across subplots + the ``with nf.style(...)`` context manager.

One colorbar serves a whole row of maps (no repetition), and the context manager
restyles otherwise-plain matplotlib code — drop it on top of anything.
"""

import nicefigs as nf
from nicefigs import data


def make():
    import cartopy.crs as ccrs

    # The context manager applies the house style just for this block.
    with nf.style("paper"):
        fig, axes = nf.subplots(
            1,
            3,
            width="full",
            panel_aspect=0.62,
            subplot_kw={"projection": ccrs.Robinson()},
            wspace=0.08,
        )
        seasons = ["DJF", "JJA", "annual"]
        m = None
        for si, ax in enumerate(axes):
            lon, lat, v = data.global_points(n=6000, field="anomaly", seed=si + 1)
            nf.draw_background(ax, ccrs)
            ax.set_extent(nf.maps.ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
            _, m = nf.binned_map(
                lon,
                lat,
                v,
                ccrs=ccrs,
                cmap="RdBu_r",
                vmin=-2.5,
                vmax=2.5,
                bin_deg=3.0,
                clip_ocean=True,
                ax=ax,
            )
        # one bar spans all three panels
        nf.colorbar(m, list(axes), label="anomaly [K]", location="bottom", thickness=0.03, shrink=0.6)
        # combined label + short title (no separate set_title → no collision)
        nf.label_panels(axes, titles=seasons)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "04_shared_colorbar", formats=("png", "pdf"))
