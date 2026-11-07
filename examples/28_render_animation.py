"""Frame-based animation engine — a multi-panel frame drawn per step, stitched.

``render_animation`` takes a ``draw(i) -> Figure`` callback, so a frame can be
anything (here: a map-like image *and* a running timeseries side by side —
which ``animate_maps`` alone can't do). Pass ``method="parallel"`` to render the
frames across CPU cores for large / expensive animations.
"""

import numpy as np

import nicefigs as nf
from nicefigs import data

_LON, _LAT, _FRAMES = data.map_timeseries_cube(n_frames=20, shape=(60, 120))
_GLOBAL_MEAN = _FRAMES.reshape(len(_FRAMES), -1).mean(axis=1)


def draw(i):
    """Draw frame ``i``: a field snapshot + the global-mean trace so far."""
    nf.set_theme("paper")
    fig, axes = nf.subplots(1, 2, width="full", panel_aspect=0.8, wspace=0.28)
    nf.image(
        _FRAMES[i],
        cmap="inferno",
        vmin=float(_FRAMES.min()),
        vmax=float(_FRAMES.max()),
        title=f"field — day {i + 1:02d}",
        ax=axes[0],
    )
    axes[1].plot(np.arange(i + 1), _GLOBAL_MEAN[: i + 1], color=nf.COLORS["blue"])
    axes[1].set_xlim(0, len(_FRAMES) - 1)
    axes[1].set_ylim(_GLOBAL_MEAN.min(), _GLOBAL_MEAN.max())
    nf.set_labels(axes[1], x="day", y="global mean")
    nf.clean_title(axes[1], "global mean")
    return fig


def make(path="28_render_animation.gif"):
    # method="sequential" here keeps the gallery self-contained; use
    # method="parallel" (the default) to fan frames out across cores.
    return nf.render_animation(draw, len(_FRAMES), path, fps=6, dpi=90, method="sequential")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    make()
