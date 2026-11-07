"""Geoscience colormap — hypsometric elevation ramp with a sea-level break."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    z = data.elevation_field()
    cmap, norm = nf.elevation_cmap(vmin=-6000, vmax=5000)
    fig, ax = nf.subplots(width="full", aspect=0.5)
    _, m = nf.image(z, cmap=cmap, norm=norm, extent=(-180, 180, -90, 90), ax=ax)
    nf.colorbar(m, ax, label="elevation [m]", extend="both")
    nf.clean_title(ax, "Synthetic topography")
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "10_elevation", formats=("png", "pdf"))
