"""Regional map — a bounded extent on a Plate Carrée axes with a slim colorbar."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    lon2d, lat2d, z = data.regional_grid(extent=(5, 20, 45, 55), res=0.08)
    fig, ax, m = nf.map(
        lon2d.ravel(),
        lat2d.ravel(),
        z.ravel(),
        extent=(5, 20, 45, 55),
        projection="platecarree",
        kind="binned",
        bin_deg=0.15,
        cmap="Spectral_r",
        inset_hist=False,
        cbar=True,
        colorbar_label="soil moisture [m³ m⁻³]",
        borders=True,
        title="Regional field (Central Europe)",
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "03_regional_map", formats=("png", "pdf"))
