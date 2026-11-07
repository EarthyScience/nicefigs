"""Global map — Robinson projection, Antarctica cropped, inset-histogram colorbar.

The house default: ``nicefigs.map`` scatters per-station values on a Robinson
world map, crops Antarctica, and draws an inset histogram that
doubles as the colorbar and shows the data distribution.
"""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    lon, lat, temp = data.global_points(field="temperature")
    fig, ax, _ = nf.map(
        lon,
        lat,
        temp,
        cmap="RdYlBu_r",
        colorbar_label="air temperature [°C]",
        title="Mean annual air temperature",
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "01_global_map", formats=("png", "pdf"))
