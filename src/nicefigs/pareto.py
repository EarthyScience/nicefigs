"""Trade-off scatter plots with the Pareto frontier drawn on top."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import matplotlib.lines as mlines
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .palettes import cmap as resolve_cmap
from .palettes import qualitative
from .scatter import MARKERS
from .style import COLORS, get_theme


def frontier(
    x: ArrayLike,
    y: ArrayLike,
    *,
    xdirection: str = "lower_better",
    ydirection: str = "higher_better",
) -> np.ndarray:
    """Indices of the Pareto-optimal points of ``(x, y)``, ordered by ``x``.

    A point is optimal iff no other point is at least as good on both axes and
    strictly better on at least one.

    Args:
        x: First objective, shape ``(n,)``.
        y: Second objective, shape ``(n,)``.
        xdirection: ``"lower_better"`` or ``"higher_better"`` for ``x``.
        ydirection: ``"lower_better"`` or ``"higher_better"`` for ``y``.

    Returns:
        Integer index array into ``x``/``y``.

    Raises:
        ValueError: A direction is not ``"lower_better"`` or ``"higher_better"``.
    """
    for d in (xdirection, ydirection):
        if d not in ("lower_better", "higher_better"):
            raise ValueError(f"a Pareto direction must be 'lower_better' or 'higher_better', got {d!r}")
    minimize_x = xdirection == "lower_better"
    maximize_y = ydirection == "higher_better"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        xb = (x <= x[i]) if minimize_x else (x >= x[i])
        yb = (y >= y[i]) if maximize_y else (y <= y[i])
        xs = (x < x[i]) if minimize_x else (x > x[i])
        ys = (y > y[i]) if maximize_y else (y < y[i])
        dominated = xb & yb & (xs | ys)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    idx = np.where(keep)[0]
    return idx[np.argsort(x[idx])]


_compute_frontier = frontier  # the `frontier` kwarg of `pareto` shadows the name


def _sizes(size, size_range, shape):
    if size is None:
        return np.full(shape, 40.0)
    s = np.asarray(size, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi <= lo:
        return np.full_like(s, float(np.mean(size_range)))
    return size_range[0] + (s - lo) / (hi - lo) * (size_range[1] - size_range[0])


def pareto(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ax: Axes | None = None,
    group: ArrayLike | None = None,
    group_colors: Mapping[Any, Any] | None = None,
    group_markers: Mapping[Any, str] | None = None,
    hue: ArrayLike | None = None,
    hue_cmap: str = "viridis",
    hue2d: tuple[ArrayLike, ArrayLike] | None = None,
    cmap2d: str = "flat",
    cmap2d_kwargs: Mapping[str, Any] | None = None,
    size: ArrayLike | None = None,
    size_range: tuple[float, float] = (15, 140),
    xdirection: str = "lower_better",
    ydirection: str = "higher_better",
    frontier: bool = True,
    frontier_per_group: bool = False,
    frontier_kwargs: Mapping[str, Any] | None = None,
    baselines: Mapping[str, float] | None = None,
    baseline_colors: Mapping[str, Any] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    xunits: str | None = None,
    yunits: str | None = None,
    edgecolor: Any = "white",
    linewidth: float = 0.6,
    legend: bool = True,
    width: float | str = "half",
) -> tuple[Axes, np.ndarray | None]:
    """Scatter a trade-off ``(x, y)`` with its Pareto frontier and optional baselines.

    Args:
        x: First objective, shape ``(n,)``.
        y: Second objective, shape ``(n,)``.
        ax: Axes to draw on; a new figure is created if ``None``.
        group: Categorical labels, shape ``(n,)``, mapped to colour and marker.
        group_colors: ``{group: colour}``; defaults to the theme palette.
        group_markers: ``{group: marker}``; defaults rotate :data:`nicefigs.scatter.MARKERS`.
        hue: Continuous values, shape ``(n,)``, coloured on ``hue_cmap`` with a
            colorbar; overrides group colours.
        hue_cmap: Colormap for ``hue``.
        hue2d: Two arrays ``(u, v)``, each shape ``(n,)``, coloured on a 2-D
            colormap; overrides ``hue`` and group colours.
        cmap2d: 2-D colormap for ``hue2d``: ``"flat"``, ``"bivariate"`` or ``"hsv"``.
        cmap2d_kwargs: Forwarded to the 2-D colormap function.
        size: Continuous values, shape ``(n,)``, mapped to marker area.
        size_range: Marker area range (pt²) that ``size`` is scaled into.
        xdirection: ``"lower_better"`` or ``"higher_better"`` for ``x``; sets the
            frontier and the arrow appended to the x-axis label.
        ydirection: As ``xdirection``, for ``y``.
        frontier: Draw the frontier line.
        frontier_per_group: Draw one colour-matched frontier per ``group``
            instead of one for all points.
        frontier_kwargs: Overrides for the frontier :meth:`~matplotlib.axes.Axes.plot` style.
        baselines: ``{label: y}`` labelled horizontal reference lines.
        baseline_colors: ``{label: colour}`` for ``baselines``; defaults to greys.
        xlabel: x-axis label.
        ylabel: y-axis label.
        xunits: Unit appended to ``xlabel``.
        yunits: Unit appended to ``ylabel``.
        edgecolor: Marker edge colour.
        linewidth: Marker edge width (pt).
        legend: Add a group legend.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.

    Returns:
        ``(ax, frontier_indices)``; the indices are those of :func:`frontier` for
        all points, or ``None`` when ``frontier`` is off or ``frontier_per_group``.

    Raises:
        KeyError: ``cmap2d`` is not a known 2-D colormap.
    """
    th = get_theme()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=0.85)

    fk = dict(color="#333333", linewidth=1.0, linestyle="--", alpha=0.7, zorder=4)
    fk.update(frontier_kwargs or {})
    sizes = _sizes(size, size_range, x.shape)

    colors2d = None
    if hue2d is not None:
        from .colormaps2d import bivariate_cmap, flat_cmap, hsv_cmap, normalize01

        u, v = hue2d
        u01 = normalize01(u)[0]
        v01 = normalize01(v)[0]
        fn = {"flat": flat_cmap, "bivariate": bivariate_cmap, "hsv": hsv_cmap}[cmap2d]
        colors2d = fn(u01, v01, **(cmap2d_kwargs or {}))

    handles = []
    mappable = None
    if group is not None:
        group = np.asarray(group)
        cats = list(dict.fromkeys(group.tolist()))
        gc = group_colors or dict(zip(cats, qualitative(len(cats))))
        gm = group_markers or {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(cats)}
        for c in cats:
            m = group == c
            ckw = dict(marker=gm[c], edgecolors=edgecolor, linewidths=linewidth, zorder=5)
            if colors2d is not None:
                ax.scatter(x[m], y[m], c=colors2d[m], s=sizes[m], **ckw)
            elif hue is not None:
                mappable = ax.scatter(
                    x[m], y[m], c=np.asarray(hue)[m], cmap=resolve_cmap(hue_cmap), s=sizes[m], **ckw
                )
            else:
                ax.scatter(x[m], y[m], color=gc[c], s=sizes[m], **ckw)
            handles.append(
                mlines.Line2D(
                    [], [], color=gc[c], marker=gm[c], linestyle="None", markersize=np.sqrt(45), label=str(c)
                )
            )
            if frontier and frontier_per_group:
                fi = _compute_frontier(x[m], y[m], xdirection=xdirection, ydirection=ydirection)
                ax.plot(x[m][fi], y[m][fi], **{**fk, "color": gc[c], "alpha": 0.85})
    else:
        ckw = dict(edgecolors=edgecolor, linewidths=linewidth, zorder=5)
        if colors2d is not None:
            ax.scatter(x, y, c=colors2d, s=sizes, **ckw)
        elif hue is not None:
            mappable = ax.scatter(x, y, c=np.asarray(hue), cmap=resolve_cmap(hue_cmap), s=sizes, **ckw)
        else:
            ax.scatter(x, y, color=COLORS["blue"], s=sizes, **ckw)

    front_idx = None
    if frontier and not frontier_per_group:
        front_idx = _compute_frontier(x, y, xdirection=xdirection, ydirection=ydirection)
        ax.plot(x[front_idx], y[front_idx], **fk)

    if baselines:
        bc = baseline_colors or {}
        fallback = ["#888888", "#AAAAAA", "#666666"]
        for i, (label, val) in enumerate(baselines.items()):
            col = bc.get(label, fallback[i % len(fallback)])
            ax.axhline(val, color=col, linestyle=":", linewidth=0.8, zorder=2)
            ax.text(
                0.995,
                val,
                f" {label} ({val:.3g})",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=th.fs_annot,
                color=col,
            )

    from .units import axis_label

    xa = axis_label(xlabel, xunits, direction=xdirection)
    ya = axis_label(ylabel, yunits, direction=ydirection)
    if xlabel:
        ax.set_xlabel(xa)
    if ylabel:
        ax.set_ylabel(ya)
    if legend and handles:
        ax.legend(handles=handles, loc="best", fontsize=th.fs_legend)
    if mappable is not None:
        from .colorbars import colorbar

        colorbar(mappable, ax)
    return ax, front_idx
