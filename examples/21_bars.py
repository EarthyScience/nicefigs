"""Grouped and stacked bars — models × metric, and land-cover composition."""

import nicefigs as nf


def make():
    nf.set_theme("paper")
    fig, axes = nf.subplots(1, 2, width="full", panel_aspect=0.8, wspace=0.32)

    regions = ["Amazon", "Congo", "Boreal", "Sahel"]
    nf.grouped_bars(
        regions,
        {"U-Net": [0.72, 0.65, 0.80, 0.55], "Transformer": [0.81, 0.74, 0.85, 0.63]},
        value_label="skill",
        rotation=30,
        ax=axes[0],
    )
    nf.stacked_bars(
        regions,
        {"forest": [0.7, 0.6, 0.5, 0.1], "grass": [0.2, 0.25, 0.2, 0.5], "bare": [0.1, 0.15, 0.3, 0.4]},
        normalize=True,
        rotation=30,
        colors={"forest": "#228833", "grass": "#CCBB44", "bare": "#BBBBBB"},
        ax=axes[1],
    )
    nf.label_panels(axes, titles=["Grouped", "Stacked composition"])
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "21_bars", formats=("png", "pdf"))
