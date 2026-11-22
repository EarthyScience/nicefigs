"""Scorecard of three models against a baseline row: text is the raw value, colour is percent better."""

import numpy as np

import nicefigs as nf


def make():
    nf.set_theme("paper")
    rows = ["CT2026", "model A", "model B", "model C"]
    cols = ["surface SH", "surface NH", "tower", "aircraft", "ship", "gradient", "OCO-2", "OCO-3", "TCCON"]
    groups = ["In situ"] * 5 + ["Gradient"] + ["Column"] * 3
    # Synthetic RMSE (ppm); smaller is better.
    rmse = np.array(
        [
            [2.10, 4.56, 6.42, 1.92, 2.97, 0.765, 0.908, 1.03, 0.898],
            [1.60, 4.78, 6.72, 2.11, 3.21, 0.926, 0.694, 0.866, 0.916],
            [1.62, 4.78, 6.75, 2.03, 2.86, 0.800, 0.691, 0.839, 0.911],
            [1.55, 4.53, 6.66, 2.05, 2.73, 0.788, 0.693, 0.841, 1.05],
        ]
    )
    # Cells whose difference to the baseline is below the noise floor of their column stay grey.
    noise = np.array([2, 1, 2, 2, 2, 6, 2, 2, 2.5])
    percent = nf.skill_gap(rmse, rmse[0], direction="lower_better", mode="percent")
    ax = nf.scorecard(
        rmse,
        row_labels=rows,
        col_labels=cols,
        direction=dict.fromkeys(cols, "lower_better"),
        reference_row="CT2026",
        unresolved=np.abs(percent) < noise,
        col_groups=groups,
        col_units=dict.fromkeys(cols, "ppm"),
        cell_format="{:#.3g}",
        colorbar=True,
    )
    return ax.figure


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    nf.save(make(), "34_scorecard_baseline", formats=("png", "pdf"))
