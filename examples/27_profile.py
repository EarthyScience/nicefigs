"""Vertical profiles — value vs pressure, inverted log axis, with a baseline."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    vp = data.vertical_profiles()
    ax = nf.profile(
        vp["profiles"],
        vp["levels"],
        baselines={"CarbonTracker": vp["baseline"]},
        xlabel="CO₂",
        xunits="ppmv",
        ylabel="pressure",
        yunits="hPa",
        log_y=True,
        title="Vertical profiles",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "27_profile", formats=("png", "pdf"))
