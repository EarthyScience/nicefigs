"""Hovmöller diagram — a latitude × time field, with model / reference / difference."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    h = data.hovmoller_field()
    diff = h["value"] - h["reference"]
    fig, _ = nf.hovmoller_panels(
        {"Model": h["value"], "Reference": h["reference"], "Difference": diff},
        h["times"],
        h["lats"],
        diff_fields={"Difference"},
        value_label="CO₂ [ppm]",
        diff_label="Δ CO₂ [ppm]",
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "26_hovmoller", formats=("png", "pdf"))
