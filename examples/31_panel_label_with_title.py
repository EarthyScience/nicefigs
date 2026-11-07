"""Panel letter + per-panel title composed onto one line.

``hovmoller_panels`` sets each panel's title via matplotlib's own
``ax.set_title`` (centred, its own default height). Calling
``label_panels(..., use_ax_title=True)`` afterwards picks that title back up,
clears it from matplotlib, and redraws it next to the bold letter at one
height — ``a  Model`` / ``b  Reference`` / ``c  Difference`` — instead of a bold
letter and a title floating at two different positions.
"""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    h = data.hovmoller_field()
    diff = h["value"] - h["reference"]
    fig, axes = nf.hovmoller_panels(
        {"Model": h["value"], "Reference": h["reference"], "Difference": diff},
        h["times"],
        h["lats"],
        diff_fields={"Difference"},
        value_label="CO₂ [ppm]",
        diff_label="Δ CO₂ [ppm]",
    )
    # Compose each panel's existing ax.set_title with the bold letter, instead
    # of letting the two render at different heights/positions.
    nf.label_panels(axes, use_ax_title=True)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "31_panel_label_with_title", formats=("png", "pdf"))
