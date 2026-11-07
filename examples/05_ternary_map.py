"""Ternary map — locations coloured by a 3-part composition, with a legend inset."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    comp, _ = data.ternary_points(n=1400)
    lon, lat, _ = data.global_points(n=1400)
    n = min(len(comp), len(lon))
    fig, ax = nf.ternary_map(
        lon[:n],
        lat[:n],
        comp[:n],
        labels=("Forest", "Crop", "Urban"),
        size=7,
        title="Dominant land use",
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "05_ternary_map", formats=("png", "pdf"))
