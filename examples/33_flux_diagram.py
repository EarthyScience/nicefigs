"""Flux diagram: a closed three-layer box budget with lateral, vertical and ground fluxes."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    b = data.box_budget()
    fig, ax = nf.subplots(width="double", aspect=0.5)
    ax, diagram = nf.flux_diagram(
        b["storage"],
        ax=ax,
        lateral=b["lateral"],
        vertical=b["vertical"],
        ground=b["ground"],
        parts=b["parts"],
        row_labels=b["row_labels"],
        col_labels=b["col_labels"],
        col_title="Region",
    )
    nf.colorbar(
        diagram.mappable, ax, location="bottom", pad=0.04, shrink=0.5, label="storage change [Tg yr⁻¹]"
    )
    nf.flux_legend(
        diagram,
        {
            "box": "box colour, upper number: storage change",
            "lateral": "lateral inflow (arrows in > 0)",
            "vertical": "upward flux (up > 0)",
            "ground": "ground flux (up > 0)",
        },
        parts={"c": "convective part", "m": "mixing part"},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "33_flux_diagram", formats=("png", "pdf"))
