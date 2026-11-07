"""Annotated heatmap — labelled rows/columns with a tight colorbar."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    fig, ax = nf.subplots(width="half", aspect=0.7)
    _, m = nf.heatmap(
        data.heatmap_matrix(shape=(6, 9)),
        row_labels=[f"model {i + 1}" for i in range(6)],
        col_labels=[f"lead {i + 1}" for i in range(9)],
        cmap="Spectral_r",
        annotate=True,
        ax=ax,
    )
    nf.colorbar(m, ax, label="skill score")
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "09_heatmap", formats=("png", "pdf"))
