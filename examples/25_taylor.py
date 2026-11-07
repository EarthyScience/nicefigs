"""Taylor diagram — correlation, standard deviation and centred RMS at a glance."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    ax = nf.taylor(data.taylor_stats(), title="Model skill vs reanalysis")
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "25_taylor", formats=("png", "pdf"))
