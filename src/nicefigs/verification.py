"""Verification diagrams for probabilistic forecasts and generative models.

:func:`reliability` plots observed against nominal coverage (or a PIT calibration
curve), :func:`rank_histogram` and :func:`rank_histogram_panels` draw Talagrand
diagrams, and :func:`precision_recall` draws precision-recall curves and points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .palettes import qualitative
from .style import COLORS, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


def reliability(
    nominal: ArrayLike | Mapping[str, ArrayLike],
    observed: ArrayLike | Mapping[str, ArrayLike],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, str] | None = None,
    markers: str | Mapping[str, str] | None = None,
    highlight: float | None = 0.90,
    highlight_color: str | None = None,
    diagonal_label: str = "Perfect calibration",
    xlabel: str = "Nominal coverage",
    ylabel: str = "Observed coverage",
    title: str = "",
    legend: bool = True,
    marker_size: float = 4.0,
    width: float | str = "half",
) -> Axes:
    """Draw a reliability diagram: observed against nominal coverage, with the 1:1 line.

    For a PIT calibration plot, pass probability levels as ``nominal`` and the
    empirical CDF of the PIT values at those levels as ``observed``.

    Args:
        nominal: Nominal levels in ``[0, 1]`` shared by all curves, or
            ``{name: levels}`` for curves evaluated at different levels.
        observed: ``{name: observed_frequency}`` aligned with ``nominal``, or one
            array for a single unnamed curve.
        ax: Axes to draw on; a new square figure is created when ``None``.
        colors: ``{name: colour}``; defaults to the theme's qualitative palette.
        markers: ``{name: marker}``, or one marker for all curves; default ``"o"``.
        highlight: Nominal level marked on the diagonal (e.g. ``0.90``); ``None``
            omits the mark.
        highlight_color: Colour of the highlight mark; default neutral grey.
        diagonal_label: Legend label of the 1:1 line; ``""`` keeps the line but
            leaves it out of the legend.
        xlabel: x-axis label.
        ylabel: y-axis label.
        title: Axes title.
        legend: Draw a legend.
        marker_size: Marker size in points.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name,
            used only when a new figure is created.

    Returns:
        The axes.
    """
    th = get_theme()
    if not isinstance(observed, dict):
        observed = {"": np.asarray(observed, dtype=float)}
    names = list(observed)
    nom_map = nominal if isinstance(nominal, dict) else None

    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=1.0)

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COLORS["neutral"], zorder=1, label=diagonal_label or None)

    if highlight is not None:
        ax.plot(
            highlight,
            highlight,
            marker="+",
            markersize=8,
            mew=1.2,
            color=highlight_color or COLORS["neutral"],
            zorder=2,
        )
        ax.axvline(highlight, color=COLORS["neutral"], lw=0.5, ls=":", alpha=0.5, zorder=0)

    cols = colors or dict(zip(names, qualitative(len(names))))
    for name in names:
        x = np.asarray(nom_map[name] if nom_map else nominal, dtype=float)
        y = np.asarray(observed[name], dtype=float)
        mk = markers.get(name) if isinstance(markers, dict) else markers
        ax.plot(
            x,
            y,
            color=cols[name],
            lw=th.lw_line,
            marker=mk or "o",
            markersize=marker_size,
            zorder=4,
            label=name or None,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    if legend and (len(names) > 1 or (names and names[0]) or diagonal_label):
        ax.legend(loc="best", fontsize=th.fs_legend)
    return ax


def rank_histogram(
    ranks: ArrayLike,
    *,
    n_ranks: int | None = None,
    ax: Axes | None = None,
    color: str | None = None,
    ideal: bool = True,
    normalize: bool = True,
    summary: tuple[str, float] | None = None,
    xlabel: str = "Rank",
    ylabel: str = "Frequency",
    title: str = "",
    width: float | str = "half",
    aspect: float = 0.72,
) -> Axes:
    """Draw a rank (Talagrand) histogram of verification ranks.

    Args:
        ranks: Integer rank per event: the 0-indexed position of the truth among
            the sorted ensemble members, in ``[0, n_members]``.
        n_ranks: Number of rank bins (``n_members + 1``); defaults to
            ``max(ranks) + 1``.
        ax: Axes to draw on; a new figure is created when ``None``.
        color: Bar colour; default the theme's blue.
        ideal: Draw the flat reference a perfectly calibrated ensemble attains.
        normalize: Show frequencies summing to 1 instead of counts.
        summary: ``(name, value)`` written in the top-left corner, e.g. a
            calibration statistic.
        xlabel: x-axis label.
        ylabel: y-axis label.
        title: Axes title.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name,
            used only when a new figure is created.
        aspect: Figure ``height / width``, used only when a new figure is created.

    Returns:
        The axes.
    """
    th = get_theme()
    ranks = np.asarray(ranks, dtype=int)
    n_ranks = n_ranks if n_ranks is not None else int(ranks.max()) + 1 if ranks.size else 1
    counts = np.bincount(ranks, minlength=n_ranks)[:n_ranks].astype(float)
    total = counts.sum()
    if normalize and total > 0:
        counts = counts / total

    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=aspect)

    x = np.arange(n_ranks)
    ax.bar(x, counts, width=0.85, color=color or COLORS["blue"], edgecolor="white", linewidth=0.4, zorder=3)

    if ideal:
        from .panels import reference_line

        flat = (1.0 / n_ranks) if normalize else total / n_ranks
        reference_line(ax, flat, axis="y", label="ideal", linestyle="--", zorder=2)

    ax.set_xlim(-0.6, n_ranks - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

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
    return ax


def rank_histogram_panels(
    ranks_by_name: Mapping[str, ArrayLike],
    *,
    n_ranks: int | None = None,
    ncols: int = 3,
    colors: Mapping[str, str] | None = None,
    ideal: bool = True,
    normalize: bool = True,
    summaries: Mapping[str, tuple[str, float]] | None = None,
    width: float | str = "full",
    panel_aspect: float = 0.8,
    titles: Sequence[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Draw one :func:`rank_histogram` panel per model as small multiples.

    Args:
        ranks_by_name: ``{name: ranks}``; see :func:`rank_histogram`.
        n_ranks: Number of rank bins shared by all panels; defaults to the
            maximum over all models, so the panels stay comparable.
        ncols: Panels per row.
        colors: ``{name: colour}``; defaults to the theme's qualitative palette.
        ideal: Draw the flat reference in each panel.
        normalize: Show frequencies instead of counts.
        summaries: ``{name: (label, value)}`` annotation per panel.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name.
        panel_aspect: Height / width of each panel.
        titles: Per-panel titles aligned with ``ranks_by_name``; defaults to the
            model names.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat array, one per model.
    """
    from .panels import facet

    names = list(ranks_by_name)
    if n_ranks is None:
        n_ranks = max(int(np.max(np.asarray(v))) for v in ranks_by_name.values() if np.size(v)) + 1
    cols = colors or dict(zip(names, qualitative(len(names))))
    summaries = summaries or {}

    def draw(ax: Axes, name: str, _i: int) -> None:
        panel_title = "" if titles is not None else name
        rank_histogram(
            ranks_by_name[name],
            n_ranks=n_ranks,
            ax=ax,
            color=cols[name],
            ideal=ideal,
            normalize=normalize,
            summary=summaries.get(name),
            title=panel_title,
        )

    return facet(names, draw, ncols=ncols, width=width, panel_aspect=panel_aspect, titles=titles)


def precision_recall(
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, str] | None = None,
    markers: str | Mapping[str, str] | None = None,
    iso_f1: Sequence[float] | None = (0.2, 0.4, 0.6, 0.8),
    diagonal: float | None = None,
    xlabel: str = "Recall",
    ylabel: str = "Precision",
    title: str = "",
    legend: bool = True,
    marker_size: float = 5.0,
    width: float | str = "half",
) -> Axes:
    """Draw a precision-recall diagram for one or more models.

    Args:
        curves: ``{name: (recall, precision)}``. Arrays draw a curve swept over a
            threshold (sorted by recall); scalars draw a single operating point.
        ax: Axes to draw on; a new square figure is created when ``None``.
        colors: ``{name: colour}``; defaults to the theme's qualitative palette.
        markers: ``{name: marker}``, or one marker for all entries. Points default
            to ``"o"``; curves have no markers unless given.
        iso_f1: F1 values drawn as dotted iso-F1 reference curves; ``None`` or
            ``()`` omits them.
        diagonal: Precision of a dashed "no skill" line (e.g. the class
            prevalence); ``None`` omits it.
        xlabel: x-axis label.
        ylabel: y-axis label.
        title: Axes title.
        legend: Draw a legend.
        marker_size: Operating-point marker size in points.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name,
            used only when a new figure is created.

    Returns:
        The axes.
    """
    th = get_theme()
    names = list(curves)
    cols = colors or dict(zip(names, qualitative(len(names))))

    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=1.0)

    if iso_f1:
        rr = np.linspace(1e-3, 1.0, 200)
        for f in iso_f1:
            with np.errstate(divide="ignore", invalid="ignore"):
                pp = f * rr / (2 * rr - f)
            valid = np.isfinite(pp) & (pp > 0) & (pp <= 1.0)
            if valid.sum() > 1:
                ax.plot(rr[valid], pp[valid], color="#cccccc", lw=0.6, ls=":", zorder=0)
                xi, yi = rr[valid][-1], pp[valid][-1]
                ax.annotate(
                    f"F1={f:g}",
                    (xi, yi),
                    fontsize=max(th.fs_annot - 0.5, 4.0),
                    color="#999999",
                    xytext=(-2, 2),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                )

    if diagonal is not None:
        ax.axhline(float(diagonal), color=COLORS["neutral"], ls="--", lw=0.7, zorder=1, label="no skill")

    for name in names:
        r, p = curves[name]
        r = np.atleast_1d(np.asarray(r, dtype=float))
        p = np.atleast_1d(np.asarray(p, dtype=float))
        mk = markers.get(name) if isinstance(markers, dict) else markers
        if r.size == 1:
            ax.plot(
                r,
                p,
                marker=mk or "o",
                color=cols[name],
                markersize=marker_size,
                linestyle="none",
                zorder=5,
                label=name,
            )
        else:
            order = np.argsort(r)
            ax.plot(
                r[order],
                p[order],
                color=cols[name],
                lw=th.lw_line,
                marker=mk,
                markersize=3.5 if mk else 0,
                zorder=4,
                label=name,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="best", fontsize=th.fs_legend)
    return ax
