"""Grouped (dodged) and stacked bar charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .figure import subplots
from .palettes import qualitative
from .style import get_theme


def _as_series(series, categories):
    """Normalise ``series`` to ``{name: array-aligned-to-categories}`` (zero-fill)."""
    out = {}
    for name, vals in series.items():
        if isinstance(vals, dict):
            out[name] = np.array([float(vals.get(c, 0.0)) for c in categories])
        else:
            out[name] = np.asarray(vals, dtype=float)
    return out


def grouped(
    categories: Sequence[Any],
    series: Mapping[str, ArrayLike | Mapping[Any, float]],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, Any] | None = None,
    hatches: Mapping[str, str] | None = None,
    width_frac: float = 0.8,
    value_label: str = "",
    value_units: str | None = None,
    baseline: float | None = None,
    legend: bool = True,
    rotation: float = 0,
    edgecolor: Any = "white",
) -> Axes:
    """Grouped (dodged) bars, one per ``(category, series)`` pair.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        categories: x-axis category labels.
        series: ``{name: values}`` with values aligned to ``categories``, or
            ``{name: {category: value}}`` where missing categories count as 0.
        colors: ``{name: colour}``; defaults to the theme palette.
        hatches: ``{name: hatch}`` as a texture encoding.
        width_frac: Fraction of each category slot covered by its bars.
        value_label: y-axis label.
        value_units: Unit appended to ``value_label``.
        baseline: y value of a dashed reference line.
        legend: Add a legend when there is more than one series.
        rotation: Category tick-label rotation (degrees).
        edgecolor: Bar edge colour.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    categories = list(categories)
    data = _as_series(series, categories)
    names = list(data)
    cols = colors or dict(zip(names, qualitative(len(names))))
    nser = len(names)
    x = np.arange(len(categories))
    bw = width_frac / max(nser, 1)

    for i, name in enumerate(names):
        offset = (i - (nser - 1) / 2) * bw
        ax.bar(
            x + offset,
            data[name],
            width=bw * 0.92,
            color=cols[name],
            edgecolor=edgecolor,
            linewidth=0.4,
            label=name,
            hatch=(hatches or {}).get(name),
        )

    if baseline is not None:
        ax.axhline(baseline, color=th.spine_color, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(
        categories, rotation=rotation, ha="right" if rotation else "center", fontsize=th.fs_tick
    )
    from .units import axis_label

    if value_label:
        ax.set_ylabel(axis_label(value_label, value_units))
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    if legend and nser > 1:
        ax.legend(loc="best", fontsize=th.fs_legend, ncol=min(nser, 3))
    return ax


def stacked(
    categories: Sequence[Any],
    segments: Mapping[str, ArrayLike | Mapping[Any, float]],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, Any] | None = None,
    width_frac: float = 0.7,
    value_label: str = "",
    value_units: str | None = None,
    legend: bool = True,
    rotation: float = 0,
    normalize: bool = False,
    edgecolor: Any = "white",
) -> Axes:
    """Stacked bars, ``segments`` stacked within each category.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        categories: x-axis category labels.
        segments: ``{name: values}`` with values aligned to ``categories``, or
            ``{name: {category: value}}`` where missing categories count as 0.
            Stacked bottom-up in mapping order.
        colors: ``{name: colour}``; defaults to the theme palette.
        width_frac: Bar width as a fraction of the category slot.
        value_label: y-axis label; defaults to ``"fraction"`` when ``normalize``.
        value_units: Unit appended to ``value_label``.
        legend: Add a legend to the right of the axes.
        rotation: Category tick-label rotation (degrees).
        normalize: Scale each category's stack to sum to 1; all-zero stacks stay 0.
        edgecolor: Bar edge colour.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    categories = list(categories)
    data = _as_series(segments, categories)
    names = list(data)
    cols = colors or dict(zip(names, qualitative(len(names))))
    x = np.arange(len(categories))
    mat = np.vstack([data[n] for n in names])  # (nseg, ncat)
    if normalize:
        totals = mat.sum(axis=0)
        mat = np.divide(mat, totals, out=np.zeros_like(mat), where=totals > 0)

    bottom = np.zeros(len(categories))
    for i, name in enumerate(names):
        ax.bar(
            x,
            mat[i],
            bottom=bottom,
            width=width_frac,
            color=cols[name],
            edgecolor=edgecolor,
            linewidth=0.4,
            label=name,
        )
        bottom += mat[i]

    ax.set_xticks(x)
    ax.set_xticklabels(
        categories, rotation=rotation, ha="right" if rotation else "center", fontsize=th.fs_tick
    )
    from .units import axis_label

    if value_label:
        ax.set_ylabel(axis_label(value_label, value_units))
    elif normalize:
        ax.set_ylabel("fraction")
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=th.fs_legend, frameon=False)
    return ax
