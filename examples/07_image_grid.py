"""Image grid — several heatmaps on a shared colour scale with one colorbar."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    imgs = [data.regional_grid(seed=i)[2] for i in range(6)]
    lim = max(abs(img).max() for img in imgs)  # signed anomalies: colour limits symmetric about zero
    fig, axes = nf.image_grid(
        imgs,
        nrows=2,
        ncols=3,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        panel_aspect=0.62,
        hspace=0.3,
        colorbar_label="anomaly [σ]",
        titles=[f"member {i + 1}" for i in range(6)],
    )
    nf.label_panels(axes)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "07_image_grid", formats=("png", "pdf"))
