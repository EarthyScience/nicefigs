"""Artist-inspired palettes — painterly categorical schemes + the Hockney base.

Swatches for a selection of MetBrewer / MoMAColors palettes and the bespoke
David-Hockney *A Bigger Grand Canyon* categorical base, plus two demos that use
it: a grouped scatter and a field coloured by the ``hockney_canyon`` colormap.
"""

import matplotlib.pyplot as plt

import nicefigs as nf
from nicefigs import data

SHOWCASE = ["nice", "hockney", "hiroshige", "klimt", "hokusai1", "monet", "vangogh2", "warhol"]


def _swatch(ax, name):
    cols = nf.artist_palette(name)
    for i, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=0.6))
    ax.set_xlim(0, max(len(cols), 1))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=nf.get_theme().fs_axis_label)


def make():
    nf.set_theme("paper")
    fig = nf.figure(width="full", aspect=0.62)
    gs = fig.add_gridspec(len(SHOWCASE), 2, width_ratios=[1.15, 1.0], wspace=0.28, hspace=0.35)

    # (a) palette swatches
    for i, name in enumerate(SHOWCASE):
        ax = fig.add_subplot(gs[i, 0])
        _swatch(ax, name)
    fig.text(0.02, 0.98, "a", fontsize=nf.get_theme().fs_panel_label, fontweight="bold", va="top")

    # (b) Hockney categorical scatter
    ax_sc = fig.add_subplot(gs[: len(SHOWCASE) // 2, 1])
    d = data.scatter_bubbles(n=150)
    nf.scatter(
        d["x"],
        d["y"],
        group=d["group"],
        group_colors=dict(zip(["A", "B", "C"], nf.qualitative(3, "hockney"))),
        size=d["size"],
        size_label="",
        legend=True,
        ax=ax_sc,
    )
    nf.clean_title(ax_sc, "Hockney categorical")
    ax_sc.set_xticks([])
    ax_sc.set_yticks([])

    # (c) Hockney continuous colormap on a field
    ax_im = fig.add_subplot(gs[len(SHOWCASE) // 2 :, 1])
    _, _, z = data.regional_grid(seed=3)
    nf.image(z, cmap="hockney_canyon", robust=True, title="hockney_canyon cmap", ax=ax_im)

    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "12_artist_palettes", formats=("svg", "png"))
