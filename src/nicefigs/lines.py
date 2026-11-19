"""Multi-series line plots against a numeric x (sweeps, thresholds, lead times), and shaded ranges."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .palettes import qualitative
from .style import get_theme
from .units import axis_label

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from numpy.typing import ArrayLike


def band(
    ax: Axes,
    x: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
    *,
    along: Literal["x", "y"] = "x",
    color: Any = "#888888",
    alpha: float = 0.16,
    zorder: float = 1.0,
    label: str | None = None,
    **kwargs: Any,
) -> PolyCollection:
    """Shade the range between ``low`` and ``high`` along a coordinate, such as a spread or an uncertainty.

    Args:
        ax: Axes to draw on.
        x: Coordinate the band runs along: the horizontal one, or with ``along="y"`` the
            vertical one, as for a profile against height or pressure.
        low: Lower bound at each ``x``; a scalar for a constant bound.
        high: Upper bound at each ``x``; a scalar for a constant bound.
        along: ``"x"`` for a band over the horizontal coordinate, ``"y"`` for one up the
            vertical coordinate.
        color: Fill colour.
        alpha: Fill opacity.
        zorder: Drawing order; below lines by default.
        label: Legend label; none by default.
        **kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.fill_between`, e.g. ``hatch``, or
            ``linewidth`` for an outline (none by default).

    Returns:
        The shaded polygon.
    """
    fill = ax.fill_between if along == "x" else ax.fill_betweenx
    kw = {"linewidth": 0, **kwargs}
    return fill(x, low, high, color=color, alpha=alpha, zorder=zorder, label=label, **kw)


def lines(
    x: ArrayLike | Mapping[str, ArrayLike],
    series: Mapping[str, ArrayLike],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, Any] | None = None,
    styles: Mapping[str, str] | None = None,
    bands: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    markers: str | Mapping[str, str] | None = None,
    logx: bool = False,
    logy: bool = False,
    logx_base: float = 10,
    baselines: Mapping[str, float] | None = None,
    baseline_colors: Mapping[str, Any] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    xunits: str | None = None,
    yunits: str | None = None,
    xdirection: str | None = None,
    ydirection: str | None = None,
    title: str = "",
    legend: bool = True,
    band_alpha: float = 0.16,
    width: float | str = "half",
    aspect: float = 0.72,
) -> Axes:
    """Plot named series against a numeric ``x``.

    Use :func:`nicefigs.timeseries` when ``x`` holds dates.

    Args:
        x: Shared x values, or ``{name: x}`` for per-series x.
        series: ``{name: y}``, one line per entry.
        ax: Axes to draw on; a new figure is created if ``None``.
        colors: ``{name: colour}``; defaults to the theme palette.
        styles: ``{name: linestyle}``, a second encoding beside colour.
        bands: ``{name: (lower, upper)}`` uncertainty shading.
        markers: One marker for all series, or ``{name: marker}``.
        logx: Log-scale the x-axis.
        logy: Log-scale the y-axis.
        logx_base: Base of the log x-axis (e.g. ``2``).
        baselines: ``{label: y}`` labelled horizontal reference lines.
        baseline_colors: ``{label: colour}`` for ``baselines``; defaults to greys.
        xlabel: x-axis label.
        ylabel: y-axis label.
        xunits: Unit appended to ``xlabel``.
        yunits: Unit appended to ``ylabel``.
        xdirection: ``"lower_better"`` or ``"higher_better"``; adds an arrow to ``xlabel``.
        ydirection: ``"lower_better"`` or ``"higher_better"``; adds an arrow to ``ylabel``.
        title: Left-aligned panel title.
        legend: Add a legend when there is more than one series.
        band_alpha: Opacity of ``bands``.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        aspect: Figure height / width when ``ax`` is ``None``.

    Returns:
        The axes.
    """
    th = get_theme()
    names = list(series)
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=aspect)
    cols = colors or dict(zip(names, qualitative(len(names))))
    styles = styles or {}
    xmap = x if isinstance(x, Mapping) else None

    for name in names:
        xs = np.asarray(xmap[name] if xmap else x, dtype=float)
        ys = np.asarray(series[name], dtype=float)
        mk = markers.get(name) if isinstance(markers, dict) else markers
        ax.plot(
            xs,
            ys,
            color=cols[name],
            linestyle=styles.get(name, "-"),
            lw=th.lw_line,
            marker=mk,
            markersize=3.5 if mk else 0,
            label=name,
            zorder=3,
        )
        if bands and name in bands:
            lo, hi = bands[name]
            band(ax, xs, lo, hi, color=cols[name], alpha=band_alpha)

    if logx:
        ax.set_xscale("log", base=logx_base)
    if logy:
        ax.set_yscale("log")

    if baselines:
        from .panels import reference_line

        bc = baseline_colors or {}
        for i, (label, val) in enumerate(baselines.items()):
            reference_line(
                ax,
                val,
                axis="y",
                label=label,
                fmt="{label} ({value:.3g})",
                color=bc.get(label, ["#888888", "#aaaaaa", "#666666"][i % 3]),
                linestyle=":",
            )

    if xlabel:
        ax.set_xlabel(axis_label(xlabel, xunits, direction=xdirection))
    if ylabel:
        ax.set_ylabel(axis_label(ylabel, yunits, direction=ydirection))
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    if legend and len(names) > 1:
        ax.legend(loc="best", fontsize=th.fs_legend)
    return ax
