"""Pareto plot — a speed vs skill trade-off, one frontier per model family."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    d = data.pareto_tradeoff()
    ax, _ = nf.pareto(
        d["x"],
        d["y"],
        group=d["group"],
        size=d["size"],
        group_colors={
            "Linear": "#4477AA",
            "Tree": "#EE6677",
            "CNN": "#228833",
            "Transformer": "#CCBB44",
        },
        frontier_per_group=True,
        xdirection="higher_better",
        ydirection="higher_better",
        xlabel="Speed",
        ylabel="Skill",
        baselines={"Persistence": 0.34, "Climatology": 0.20},
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "18_pareto", formats=("png", "pdf"))
