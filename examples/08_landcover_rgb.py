"""RGB imagery + categorical land cover — a geospecific palette with a legend."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    fig, axes = nf.subplots(1, 2, width="half", panel_aspect=1.0, wspace=0.08)
    nf.image(data.rgb_patch(), rgb=True, stretch=True, ax=axes[0])

    lc = nf.categorical("esa_worldcover")
    nf.image(lc.to_rgb(data.landcover_patch()), rgb=True, ax=axes[1])
    axes[1].legend(
        handles=lc.legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=nf.get_theme().fs_annot,
        frameon=False,
        handlelength=1.0,
        labelspacing=0.35,
    )
    # combined label + title (top-left) avoids clashing with a centred title
    nf.label_panels(axes, titles=["Sentinel-2 RGB", "ESA WorldCover"])
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "08_landcover_rgb", formats=("png", "pdf"))
