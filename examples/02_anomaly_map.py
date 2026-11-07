"""Binned anomaly map — diverging colormap, ocean clipped, symmetric limits.

For dense fields, ``kind="binned"`` bin-averages onto a regular grid and masks
the ocean. ``diverging=True`` picks a diverging colormap and symmetric limits
about zero — the right default for anomalies.
"""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    lon, lat, anom = data.global_points(n=9000, field="anomaly")
    fig, ax, _ = nf.map(
        lon,
        lat,
        anom,
        kind="binned",
        diverging=True,
        colorbar_label="temperature anomaly [K]",
        title="Temperature anomaly",
        bin_deg=2.5,
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "02_anomaly_map", formats=("png", "pdf"))
