"""Agreement plot — KDE density, 1:1 line, outlier scatter, regression, metrics."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    obs, pred = data.xy_agreement(r=0.92)
    ax, stats = nf.agreement(
        obs,
        pred,
        xlabel="observed runoff",
        ylabel="predicted runoff",
        units="mm d⁻¹",
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "14_agreement", formats=("png", "pdf"))
