"""Cross-section / curtain — latitude × pressure, value and anomaly panels."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    cs = data.cross_section_field()
    val = cs["value"]
    anom = val - val.mean(axis=0, keepdims=True)
    fig, axes = nf.cross_section_panels(
        {"CO₂": val, "anomaly": anom},
        cs["lats"],
        cs["levels"],
        diff_fields={"anomaly"},
        value_label="CO₂ [ppm]",
        diff_label="Δ CO₂ [ppm]",
        log_y=True,
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "22_cross_section", formats=("png", "pdf"))
