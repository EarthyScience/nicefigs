"""Composite publication figure — maps, timeseries and agreement in one layout.

Shows the pieces composing: a GridSpec from ``nicefigs.grid`` (a projected map
panel plus two analysis panels), a colorbar hugging the map, and ``(a) (b) (c)``
panel labels — all at a single physical width so fonts stay consistent.
"""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    fig = nf.figure(width="full", aspect=0.62)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.32)

    # (a) global map spanning the top row
    ax_map, ccrs = nf.make_map_axes(fig, gs[0, :], projection="robinson")
    lon, lat, temp = data.global_points(field="temperature")
    _, m = nf.scatter_map(lon, lat, temp, ccrs=ccrs, cmap="RdYlBu_r", size=4, ax=ax_map)
    nf.draw_background(ax_map, ccrs)
    ax_map.set_extent(nf.maps.ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
    nf.colorbar(m, ax_map, label="T [°C]")

    # (b) timeseries
    ax_ts = fig.add_subplot(gs[1, 0])
    dates, series = data.timeseries(n_days=730)
    nf.timeseries(dates, series, ax=ax_ts, highlight="observed", ylabel="T", yunits="degC", legend=False)
    ax_ts.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    # (c) agreement
    ax_ag = fig.add_subplot(gs[1, 1])
    obs, pred = data.xy_agreement()
    nf.agreement(obs, pred, ax=ax_ag, xlabel="obs", ylabel="pred", units="mm")

    nf.label_panels([ax_map, ax_ts, ax_ag])
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "23_composite_figure", formats=("png", "pdf"))
