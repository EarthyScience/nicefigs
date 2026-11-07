"""Model comparison — one panel per metric, a violin per model, split by timescale."""

import nicefigs as nf
from nicefigs import data


def make():
    nf.set_theme("paper")
    scores = data.model_scores()
    fig, axes = nf.model_comparison(
        scores,
        group_label="timescale",
        kind="violin",
        metric_units={"RMSE": "mm"},
    )
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "17_model_comparison", formats=("png", "pdf"))
