"""Caterpillar (forest) plots: per-item values with error bars, ranked."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .palettes import qualitative
from .style import COLORS, get_theme


def _resolve_bounds(values, err, lo, hi):
    values = np.asarray(values, dtype=float)
    if lo is not None and hi is not None:
        return values, np.asarray(lo, float), np.asarray(hi, float)
    if err is not None:
        err = np.asarray(err, dtype=float)
        return values, values - err, values + err
    return values, values, values


def caterpillar(
    labels: Sequence[Any],
    values: ArrayLike,
    *,
    err: ArrayLike | None = None,
    lo: ArrayLike | None = None,
    hi: ArrayLike | None = None,
    ax: Axes | None = None,
    sort_by: str | ArrayLike = "value",
    ascending: bool = True,
    orientation: str = "horizontal",
    groups: ArrayLike | None = None,
    group_colors: Mapping[Any, Any] | None = None,
    color: Any = None,
    baseline: ArrayLike | tuple[ArrayLike, ArrayLike] | None = None,
    baseline_label: str = "baseline",
    baseline_color: Any = COLORS["neutral"],
    offset: float = 0.18,
    ref_line: float | None = 0.0,
    ref_label: str = "",
    summary: tuple[str, float] | None = None,
    value_label: str = "",
    value_units: str | None = None,
    annotate_threshold: float | None = None,
    markersize: float = 4,
    capsize: float = 1.5,
    width: float | str = "half",
    aspect: float | None = None,
    legend: bool = True,
) -> Axes:
    """Ranked per-item value ± error plot.

    Args:
        labels: Item names, one per value.
        values: Central values, shape ``(n,)``.
        err: Symmetric error, shape ``(n,)``; ignored when ``lo`` and ``hi`` are given.
        lo: Explicit lower bounds, shape ``(n,)``.
        hi: Explicit upper bounds, shape ``(n,)``.
        ax: Axes to draw on; a new figure is created if ``None``.
        sort_by: ``"value"`` (rank by value), ``"none"`` (input order), or an
            array of sort keys, shape ``(n,)``.
        ascending: Sort order.
        orientation: ``"horizontal"`` (items down y, the forest layout) or
            ``"vertical"`` (items along x).
        groups: Per-item categorical labels, shape ``(n,)``, mapped to colour
            with a legend.
        group_colors: ``{group: colour}``; defaults to the theme palette.
        color: Point colour without ``groups``.
        baseline: A second series, ``values`` or ``(values, err)``, drawn offset
            beside each item.
        baseline_label: Legend label of ``baseline``.
        baseline_color: Colour of ``baseline``.
        offset: Offset between the series and ``baseline``, in item slots.
        ref_line: Reference line value; ``None`` omits it.
        ref_label: Text placed in the corner to name ``ref_line``.
        summary: ``(name, value)`` shown as a headline annotation.
        value_label: Value-axis label.
        value_units: Unit appended to ``value_label``.
        annotate_threshold: Label items whose ``|value|`` is at least this.
        markersize: Point size (pt).
        capsize: Error-bar cap length (pt).
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        aspect: Figure height / width when ``ax`` is ``None``; scales with the
            item count by default.
        legend: Add a legend when ``groups`` or ``baseline`` is set.

    Returns:
        The axes.
    """
    th = get_theme()
    labels = list(labels)
    values, low, high = _resolve_bounds(values, err, lo, hi)
    n = len(labels)

    if isinstance(sort_by, str):
        if sort_by == "value":
            order = np.argsort(values)
            if not ascending:
                order = order[::-1]
        else:
            order = np.arange(n)
    else:
        order = np.argsort(np.asarray(sort_by))
        if not ascending:
            order = order[::-1]

    labels = [labels[i] for i in order]
    values, low, high = values[order], low[order], high[order]
    lo_err = np.clip(values - low, 0, None)
    hi_err = np.clip(high - values, 0, None)
    pos = np.arange(n)

    if ax is None:
        from .figure import subplots

        asp = aspect if aspect is not None else max(0.35, 0.06 * n)
        _, ax = subplots(width=width, aspect=asp)

    if groups is not None:
        groups = np.asarray(groups)[order]
        cats = list(dict.fromkeys(groups.tolist()))
        gc = group_colors or dict(zip(cats, qualitative(len(cats))))
        colors = [gc[g] for g in groups]
    else:
        colors = color or COLORS["blue"]

    horiz = orientation == "horizontal"

    def _errbar(p, v, le, he, c, **kw):
        if horiz:
            ax.errorbar(
                v,
                p,
                xerr=[le, he],
                fmt="o",
                color=c,
                ecolor=c,
                markersize=markersize,
                capsize=capsize,
                elinewidth=0.8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                **kw,
            )
        else:
            ax.errorbar(
                p,
                v,
                yerr=[le, he],
                fmt="o",
                color=c,
                ecolor=c,
                markersize=markersize,
                capsize=capsize,
                elinewidth=0.8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                **kw,
            )

    off = offset if baseline is not None else 0.0
    if groups is not None:
        for c in cats:
            m = np.array([g == c for g in groups])
            _errbar(pos[m] + off, values[m], lo_err[m], hi_err[m], gc[c], zorder=5, label=str(c))
    else:
        _errbar(pos + off, values, lo_err, hi_err, colors, zorder=5)

    if baseline is not None:
        bvals, berr, blo, bhi = _baseline_arrays(baseline, order)
        ble = np.clip(bvals - blo, 0, None)
        bhe = np.clip(bhi - bvals, 0, None)
        _errbar(pos - off, bvals, ble, bhe, baseline_color, zorder=4, alpha=0.8, label=baseline_label)

    if ref_line is not None:
        (ax.axvline if horiz else ax.axhline)(
            ref_line, color=COLORS["neutral"], linestyle="--", linewidth=0.7, zorder=1
        )
        if ref_label:
            ax.text(
                0.99,
                0.99,
                ref_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=th.fs_annot,
                color=COLORS["neutral"],
            )

    from .units import axis_label

    vlabel = axis_label(value_label, value_units) if value_label else ""
    if horiz:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=th.fs_tick)
        ax.set_ylim(-0.6, n - 0.4)
        if vlabel:
            ax.set_xlabel(vlabel)
        ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=th.fs_tick)
        ax.set_xlim(-0.6, n - 0.4)
        if vlabel:
            ax.set_ylabel(vlabel)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    if annotate_threshold is not None:
        for p, v, lab in zip(pos, values, labels):
            if abs(v) >= annotate_threshold:
                xy, xytext, ha, va = (
                    ((v, p), (3, 0), "left", "center") if horiz else ((p, v), (0, 3), "center", "bottom")
                )
                ax.annotate(
                    lab, xy, fontsize=th.fs_annot, xytext=xytext, textcoords="offset points", ha=ha, va=va
                )

    if summary is not None:
        name, val = summary
        ax.text(
            0.02,
            0.98,
            f"{name} = {val:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=th.fs_annot,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.4, alpha=0.85),
        )

    if legend and (groups is not None or baseline is not None):
        ax.legend(loc="best", fontsize=th.fs_legend)
    return ax


def _baseline_arrays(baseline, order):
    if isinstance(baseline, tuple) and len(baseline) == 2:
        bvals, berr = baseline
        bvals = np.asarray(bvals, float)[order]
        berr = np.asarray(berr, float)[order]
        return bvals, berr, bvals - berr, bvals + berr
    bvals = np.asarray(baseline, float)[order]
    return bvals, None, bvals, bvals
