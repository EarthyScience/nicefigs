"""Scatter plots that encode extra variables as marker shape, size and colour."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.lines as mlines
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from numpy.typing import ArrayLike

from matplotlib.markers import MarkerStyle

from .figure import subplots
from .palettes import cmap as resolve_cmap
from .palettes import qualitative
from .style import get_theme

#: Marker rotation for categorical groups.
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def _scale_size(size, size_range):
    s = np.asarray(size, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi <= lo:
        return np.full_like(s, np.mean(size_range))
    return size_range[0] + (s - lo) / (hi - lo) * (size_range[1] - size_range[0])


def scatter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ax: Axes | None = None,
    color: Any = None,
    hue: ArrayLike | None = None,
    hue_cmap: str = "viridis",
    hue_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    size: ArrayLike | None = None,
    size_range: tuple[float, float] = (8, 120),
    size_label: str = "",
    group: ArrayLike | None = None,
    group_markers: Mapping[Any, str] | None = None,
    group_colors: Mapping[Any, Any] | None = None,
    marker: str = "o",
    alpha: float = 0.85,
    edgecolor: Any = "none",
    linewidth: float = 0.0,
    xerr: ArrayLike | None = None,
    yerr: ArrayLike | None = None,
    point_labels: Sequence[Any] | None = None,
    label_top: int = 0,
    legend: bool = True,
    colorbar: bool = False,
    **kwargs: Any,
) -> tuple[Axes, PathCollection | dict[Any, PathCollection]]:
    """Scatter ``y`` against ``x`` with optional shape, size and colour encodings.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        x: Point x coordinates, shape ``(n,)``.
        y: Point y coordinates, shape ``(n,)``.
        color: Single colour for all points; ignored when ``hue`` or ``group`` is set.
        hue: Continuous values, shape ``(n,)``, mapped to colour on ``hue_cmap``.
        hue_cmap: Colormap name for ``hue``.
        hue_label: Colorbar label for ``hue``.
        vmin: Lower ``hue`` colour limit.
        vmax: Upper ``hue`` colour limit.
        size: Continuous values, shape ``(n,)``, mapped to marker area.
        size_range: Marker area range (pt²) that ``size`` is scaled into.
        size_label: Title of the size legend.
        group: Categorical labels, shape ``(n,)``, mapped to marker shape and,
            unless ``hue`` is set, colour.
        group_markers: ``{group: marker}`` overrides; defaults rotate :data:`MARKERS`.
        group_colors: ``{group: colour}`` overrides; defaults to the theme palette.
        marker: Marker shape when ``group`` is not set.
        alpha: Marker opacity.
        edgecolor: Marker edge colour.
        linewidth: Marker edge width (pt).
        xerr: Per-point x error bars, drawn faint behind the markers.
        yerr: Per-point y error bars, drawn faint behind the markers.
        point_labels: Per-point text labels, shape ``(n,)``.
        label_top: Annotate only the ``label_top`` points furthest from the
            centroid with ``point_labels``; ``0`` annotates none.
        legend: Add group and size legends where they apply.
        colorbar: Add a colorbar for ``hue``.
        **kwargs: Forwarded to :meth:`matplotlib.axes.Axes.scatter`.

    Returns:
        ``(ax, artists)``: the axes and the scatter collection, or ``{group: collection}`` when ``group`` is given.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = _scale_size(size, size_range) if size is not None else None
    s_default = kwargs.pop("s", 36)

    if xerr is not None or yerr is not None:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#999999",
            elinewidth=0.6,
            capsize=1.2,
            zorder=1,
            alpha=0.7,
        )

    handles = []
    mappable = None

    if group is not None:
        group = np.asarray(group)
        cats = list(dict.fromkeys(group.tolist()))
        gm = group_markers or {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(cats)}
        gc = group_colors or (dict(zip(cats, qualitative(len(cats)))) if hue is None else None)
        colls = {}
        for c in cats:
            m = group == c
            kwc = dict(marker=gm[c], alpha=alpha, edgecolors=edgecolor, linewidths=linewidth)
            kwc["s"] = s[m] if s is not None else s_default
            if hue is not None:
                kwc.update(c=np.asarray(hue)[m], cmap=resolve_cmap(hue_cmap), vmin=vmin, vmax=vmax)
            else:
                kwc["color"] = gc[c]
            colls[c] = ax.scatter(x[m], y[m], **kwc, **kwargs)
            mappable = colls[c]
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=(gc[c] if gc else th.text_color),
                    marker=gm[c],
                    linestyle="None",
                    markersize=np.sqrt(40),
                    label=str(c),
                )
            )
        result = colls
    else:
        kwc = dict(marker=marker, alpha=alpha, linewidths=linewidth)
        if MarkerStyle(marker).is_filled():  # an unfilled marker ("x", "+") is drawn by its edge colour
            kwc["edgecolors"] = edgecolor
        kwc["s"] = s if s is not None else s_default
        if hue is not None:
            kwc.update(c=np.asarray(hue, dtype=float), cmap=resolve_cmap(hue_cmap), vmin=vmin, vmax=vmax)
        else:
            kwc["color"] = color or th.palette[0]
        mappable = ax.scatter(x, y, **kwc, **kwargs)
        result = mappable

    if legend and group is not None:
        leg = ax.legend(
            handles=handles, loc="best", fontsize=th.fs_legend, title=None, handletextpad=0.4, borderpad=0.4
        )
        ax.add_artist(leg)

    if legend and size is not None:
        _add_size_legend(ax, size, size_range, size_label, th)

    if colorbar and hue is not None and mappable is not None:
        from .colorbars import colorbar as _cb

        _cb(mappable, ax, label=hue_label)

    if point_labels is not None and label_top > 0:
        d = np.hypot(x - np.nanmean(x), y - np.nanmean(y))
        for i in np.argsort(d)[::-1][:label_top]:
            ax.annotate(
                str(point_labels[i]),
                (x[i], y[i]),
                fontsize=th.fs_annot,
                xytext=(3, 2),
                textcoords="offset points",
            )

    return ax, result


def _add_size_legend(ax, size, size_range, label, th):
    s = np.asarray(size, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi <= lo:
        return
    levels = np.linspace(lo, hi, 3)
    sizes = _scale_size(levels, size_range)
    handles = [
        mlines.Line2D(
            [],
            [],
            color="#666666",
            marker="o",
            linestyle="None",
            markersize=np.sqrt(sz),
            markerfacecolor="#bbbbbb",
            markeredgecolor="#666666",
            label=f"{lv:.3g}",
        )
        for lv, sz in zip(levels, sizes)
    ]
    # The group legend is kept via add_artist; adding this one too would render it twice.
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=th.fs_legend,
        title=label or None,
        title_fontsize=th.fs_legend,
        labelspacing=1.2,
        handletextpad=0.6,
        borderpad=0.5,
        frameon=False,
    )
