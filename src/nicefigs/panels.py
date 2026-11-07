"""Multi-panel layout: grids, small multiples, panel letters, row/column headers and titles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .figure import figure_size
from .style import Theme, get_theme, style


def _is_geoaxes(ax: Axes) -> bool:
    return hasattr(ax, "projection") and ax.__class__.__name__ in ("GeoAxes", "GeoAxesSubplot")


def _dedupe_title(text: str, label: str, title: str) -> str:
    """Strip a leading ``text`` or ``label`` followed by whitespace (or the whole string) from ``title``."""
    stripped = title.lstrip()
    for prefix in (text, label):
        if not prefix:
            continue
        if stripped == prefix:
            return ""
        if stripped.startswith(prefix) and stripped[len(prefix)].isspace():
            return stripped[len(prefix) :].lstrip()
    return title


def panel_label(
    ax: Axes,
    label: str,
    title: str = "",
    *,
    x: float = 0.0,
    y: float = 1.0,
    pad: float | None = None,
    weight: str = "bold",
    parens: bool = False,
    use_ax_title: bool = False,
    dedupe: bool = False,
) -> None:
    """Draw a panel label (``a``) and an optional short title above ``ax``.

    The label is bold lowercase without parentheses, set at the theme's
    ``fs_panel_label`` (1.2 times the axis-label size). Its left edge is flush
    with the axes' left edge and its baseline sits ``pad`` points above the top
    edge. The title follows on the same baseline in normal weight at
    ``fs_panel_title``, half a label-height after the label. On cartopy map
    axes both are anchored to the axes' grid cell, so labels line up down the
    figure.

    Note:
        The title is drawn as passed: label ``"a"`` with title
        ``"a priori flux"`` renders ``a  a priori flux``. Pass ``dedupe=True``
        to strip a title prefix equal to the label.

    Args:
        ax: Target axes.
        label: Label text, e.g. ``"a"``.
        title: Short title drawn after the label.
        x: Anchor x-position in axes fraction (of the grid cell for map axes).
        y: Anchor y-position, same coordinates as ``x``.
        pad: Height of the baseline above the anchor (pt); defaults to half the
            label size.
        weight: Font weight of the label; the title is always normal weight.
        parens: Wrap the label in parentheses, e.g. ``"(a)"``.
        use_ax_title: When ``title`` is empty, use the axes' own left, centre
            or right title instead and clear it, so it is not drawn twice.
        dedupe: Strip a leading copy of the label (and whitespace) from ``title``.
    """
    from matplotlib.transforms import offset_copy

    th = get_theme()
    text = f"({label})" if parens else label
    if not title and use_ax_title:
        for loc in ("left", "center", "right"):
            title = ax.get_title(loc=loc)
            if title:
                ax.set_title("", loc=loc)
                break
    if title and dedupe:
        title = _dedupe_title(text, label, title)
    pad = 0.5 * th.fs_panel_label if pad is None else pad
    fig = ax.figure
    if _is_geoaxes(ax):
        # The original (pre-aspect) position is the grid cell, not the pillar-boxed map.
        fig.canvas.draw()
        pos = ax.get_position(original=True)
        base, target = fig.transFigure, fig
        tx, ty = pos.x0 + x * pos.width, pos.y1 + (y - 1.0) * pos.height
    else:
        base, target = ax.transAxes, ax
        tx, ty = x, y
    at_label = offset_copy(base, fig=fig, x=0.0, y=pad, units="points")
    mark = target.text(
        tx,
        ty,
        text,
        transform=at_label,
        fontsize=th.fs_panel_label,
        fontweight=weight,
        va="baseline",
        ha="left",
    )
    if title:
        width_pt = mark.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi
        gap_pt = width_pt + 0.5 * th.fs_panel_label
        at_title = offset_copy(base, fig=fig, x=gap_pt, y=pad, units="points")
        target.text(tx, ty, title, transform=at_title, fontsize=th.fs_panel_title, va="baseline", ha="left")


def label_panels(
    axes: Iterable[Axes | None],
    titles: Sequence[str] | None = None,
    *,
    start: int = 0,
    upper: bool = False,
    parens: bool = False,
    x: float = 0.0,
    y: float = 1.0,
    pad: float | None = None,
    skip_none: bool = True,
    use_ax_title: bool = False,
    dedupe: bool = False,
) -> None:
    """Label axes ``a, b, c, …`` in reading order with :func:`panel_label`.

    Args:
        axes: Axes, or an array of them (flattened row-major).
        titles: Per-panel short titles, aligned with the flattened ``axes``.
        start: Offset into the alphabet (``0`` is ``a``).
        upper: Use ``A, B, C``.
        parens: Wrap the letter in parentheses.
        x: Anchor x-position in axes fraction.
        y: Anchor y-position in axes fraction.
        pad: Height of the label baseline above the anchor (pt); see :func:`panel_label`.
        skip_none: Skip ``None`` entries (empty grid cells) without using a letter.
        use_ax_title: For panels with no entry in ``titles``, fall back to
            each axes' own ``ax.get_title()`` and clear it — see
            :func:`panel_label`.
        dedupe: Drop a leading duplicate of the letter from the title — see
            :func:`panel_label`.
    """
    flat = (
        list(np.asarray(list(axes), dtype=object).ravel())
        if not isinstance(axes, np.ndarray)
        else list(axes.ravel())
    )
    i = 0
    for k, ax in enumerate(flat):
        if ax is None and skip_none:
            continue
        letter = chr(ord("A" if upper else "a") + start + i)
        title = titles[k] if titles is not None and k < len(titles) else ""
        panel_label(
            ax,
            letter,
            title or "",
            x=x,
            y=y,
            pad=pad,
            parens=parens,
            use_ax_title=use_ax_title,
            dedupe=dedupe,
        )
        i += 1


def _cell_axes(axes: Any, which: str) -> list[Axes | None]:
    """Pick the labelling axes: a 2-D grid's first column/row, or a 1-D list."""
    arr = np.asarray(axes, dtype=object)
    if arr.ndim == 2:
        return list(arr[:, 0]) if which == "row" else list(arr[0, :])
    return list(arr.ravel())


def row_labels(
    axes: Any,
    labels: Sequence[str],
    *,
    x: float = -0.22,
    rotation: float = 90.0,
    fontsize: float | None = None,
    weight: str = "normal",
    pad: float = 0.0,
) -> None:
    """Label the rows of a panel grid with rotated headers down the left margin.

    Args:
        axes: A ``(nrows, ncols)`` grid (its first column is used), or one axes per row.
        labels: One header per row; empty strings are skipped.
        x: Header x-position in axes fraction.
        rotation: Text rotation (degrees).
        fontsize: Font size (pt); defaults to the theme's ``fs_panel_title``.
        weight: Font weight.
        pad: Vertical shift of the header from the row centre, in axes fraction.
    """
    th = get_theme()
    fs = fontsize if fontsize is not None else th.fs_panel_title
    targets = _cell_axes(axes, "row")
    for ax, lab in zip(targets, labels):
        if ax is None or not lab:
            continue
        if _is_geoaxes(ax):
            fig = ax.figure
            fig.canvas.draw()
            pos = ax.get_position(original=True)
            fig.text(
                pos.x0 + x * pos.width,
                pos.y0 + 0.5 * pos.height,
                lab,
                rotation=rotation,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=weight,
            )
        else:
            ax.text(
                x,
                0.5 + pad,
                lab,
                transform=ax.transAxes,
                rotation=rotation,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=weight,
            )


def col_labels(
    axes: Any,
    labels: Sequence[str],
    *,
    y: float = 1.05,
    fontsize: float | None = None,
    weight: str = "normal",
    pad: float = 0.0,
) -> None:
    """Label the columns of a panel grid with headers above the top row.

    Args:
        axes: A ``(nrows, ncols)`` grid (its first row is used), or one axes per column.
        labels: One header per column; empty strings are skipped.
        y: Header y-position in axes fraction.
        fontsize: Font size (pt); defaults to the theme's ``fs_panel_title``.
        weight: Font weight.
        pad: Horizontal shift of the header from the column centre, in axes fraction.
    """
    th = get_theme()
    fs = fontsize if fontsize is not None else th.fs_panel_title
    targets = _cell_axes(axes, "col")
    for ax, lab in zip(targets, labels):
        if ax is None or not lab:
            continue
        if _is_geoaxes(ax):
            fig = ax.figure
            fig.canvas.draw()
            pos = ax.get_position(original=True)
            fig.text(
                pos.x0 + 0.5 * pos.width,
                pos.y1 + (y - 1.0) * pos.height,
                lab,
                ha="center",
                va="bottom",
                fontsize=fs,
                fontweight=weight,
            )
        else:
            ax.text(
                0.5 + pad,
                y,
                lab,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=fs,
                fontweight=weight,
            )


def clean_title(ax: Axes, title: str, *, loc: str = "left", pad: float = 3.0) -> None:
    """Set an axes title at the theme's title size, left-aligned by default.

    Args:
        ax: Target axes.
        title: Title text.
        loc: ``"left"``, ``"center"`` or ``"right"``.
        pad: Gap between title and axes (pt).
    """
    th = get_theme()
    ax.set_title(title, fontsize=th.fs_title, loc=loc, pad=pad)


def despine(
    ax: Axes, *, top: bool = True, right: bool = True, left: bool = False, bottom: bool = False
) -> None:
    """Hide the selected spines of ``ax`` (top and right by default)."""
    for side, off in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        if off:
            ax.spines[side].set_visible(False)


def grid(
    nrows: int,
    ncols: int,
    *,
    width: float | str | None = None,
    height: float | None = None,
    aspect: float | None = None,
    panel_aspect: float | None = None,
    theme: str | Theme | None = None,
    hspace: float = 0.28,
    wspace: float = 0.22,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    projection: Any = None,
    with_legend_row: bool = False,
    legend_row_height: float = 0.12,
) -> tuple[Figure, np.ndarray] | tuple[Figure, np.ndarray, Axes]:
    """Create a GridSpec panel grid at a physical size.

    Args:
        nrows: Panel rows.
        ncols: Panel columns.
        width: Inches or a preset name; see :func:`nicefigs.figure_size`.
        height: Inches.
        aspect: Whole-figure ``height / width``.
        panel_aspect: Per-panel ``height / width``; overrides ``aspect``.
        theme: Theme applied while the figure is created; ``None`` uses the active theme.
        hspace: Vertical gap between panels (fraction of mean panel height).
        wspace: Horizontal gap between panels (fraction of mean panel width).
        width_ratios: Relative column widths.
        height_ratios: Relative row heights.
        projection: Projection for every panel, e.g. a cartopy CRS.
        with_legend_row: Add a hidden, full-width bottom row for a shared legend.
        legend_row_height: Height ratio of the legend row relative to a panel row.

    Returns:
        ``(fig, axes)`` with ``axes`` a ``(nrows, ncols)`` object array, or
        ``(fig, axes, ax_legend)`` when ``with_legend_row`` is set.
    """
    from matplotlib.gridspec import GridSpec

    size = figure_size(width, height, aspect, nrows=nrows, ncols=ncols, panel_aspect=panel_aspect)
    th = get_theme(theme)

    total_rows = nrows + (1 if with_legend_row else 0)
    hr = list(height_ratios) if height_ratios is not None else [1.0] * nrows
    if with_legend_row:
        hr = hr + [legend_row_height]
    with style(th):
        fig = plt.figure(figsize=size, dpi=th.dpi)
        gs = GridSpec(
            total_rows,
            ncols,
            figure=fig,
            hspace=hspace,
            wspace=wspace,
            width_ratios=width_ratios,
            height_ratios=hr,
        )
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j], projection=projection)
        if with_legend_row:
            ax_legend = fig.add_subplot(gs[nrows, :])
            ax_legend.axis("off")
            return fig, axes, ax_legend
    return fig, axes


def reference_line(
    ax: Axes,
    value: float,
    *,
    axis: str = "y",
    label: str = "",
    color: str | None = None,
    linestyle: str = "--",
    linewidth: float = 0.7,
    alpha: float = 0.7,
    fmt: str = "{label}",
    zorder: float = 1,
) -> None:
    """Draw a horizontal (``axis="y"``) or vertical (``axis="x"``) reference line with an optional label.

    Args:
        ax: Target axes.
        value: Position of the line in data coordinates.
        axis: ``"y"`` for a horizontal line at ``y=value``, ``"x"`` for a vertical one.
        label: Text drawn at the line's far end; empty draws none.
        color: Line and label colour; defaults to grey.
        linestyle: Line style.
        linewidth: Line width (pt).
        alpha: Line opacity.
        fmt: Label format with ``{label}`` and ``{value}`` fields, e.g. ``"{label} ({value:.2g})"``.
        zorder: Line drawing order.
    """
    th = get_theme()
    color = color or "#888888"
    (ax.axhline if axis == "y" else ax.axvline)(
        value, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder
    )
    if label:
        text = fmt.format(label=label, value=value)
        if axis == "y":
            ax.text(
                0.995,
                value,
                f" {text}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=th.fs_annot,
                color=color,
            )
        else:
            ax.text(
                value,
                0.995,
                f" {text}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                rotation=90,
                fontsize=th.fs_annot,
                color=color,
            )


def facet(
    items: Iterable[Any],
    draw: Callable[[Axes, Any, int], object],
    *,
    ncols: int = 3,
    width: float | str = "full",
    panel_aspect: float = 0.8,
    theme: str | Theme | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    hide_unused: bool = True,
    labels: bool = False,
    titles: Sequence[str] | None = None,
    hspace: float = 0.35,
    wspace: float = 0.22,
) -> tuple[Figure, np.ndarray]:
    """Draw small multiples: one panel per item, filled by ``draw(ax, item, i)``.

    Args:
        items: One entry per panel, passed to ``draw``.
        draw: Callback drawing ``item`` (the ``i``-th) into ``ax``.
        ncols: Maximum panel columns; rows follow from ``len(items)``.
        width: Figure width, inches or a preset name.
        panel_aspect: Per-panel ``height / width``.
        theme: Theme applied while the figure is created; ``None`` uses the active theme.
        sharex: Forwarded to :func:`matplotlib.pyplot.subplots`.
        sharey: Forwarded to :func:`matplotlib.pyplot.subplots`.
        hide_unused: Hide the trailing cells without an item.
        labels: Add ``a, b, c`` panel labels.
        titles: Per-panel titles set with :func:`clean_title`.
        hspace: Vertical gap between panels.
        wspace: Horizontal gap between panels.

    Returns:
        ``(fig, axes)`` with ``axes`` the flat array of all cells, unused ones included.
    """
    items = list(items)
    n = len(items)
    ncols = min(ncols, n) if n else ncols
    nrows = int(np.ceil(n / ncols)) if n else 1

    size = figure_size(width, panel_aspect=panel_aspect, nrows=nrows, ncols=ncols)
    th = get_theme(theme)
    with style(th):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=size,
            dpi=th.dpi,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=dict(hspace=hspace, wspace=wspace),
        )
    axes_flat = axes.ravel()
    for i, (ax, item) in enumerate(zip(axes_flat, items)):
        draw(ax, item, i)
        if titles is not None and i < len(titles):
            clean_title(ax, titles[i])
    if hide_unused:
        for ax in axes_flat[n:]:
            ax.axis("off")
    if labels:
        label_panels(axes_flat[:n])
    return fig, axes_flat
