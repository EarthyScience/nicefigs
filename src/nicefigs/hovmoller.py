"""Hovmöller diagrams: a field over time and one spatial coordinate."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .palettes import cmap as resolve_cmap
from .palettes import robust_limits, symmetric_limits
from .style import get_theme
from .units import axis_label

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


def _is_dates(x: ArrayLike) -> bool:
    import datetime as _dt

    arr = list(x)
    return bool(arr) and isinstance(arr[0], (_dt.date, _dt.datetime, np.datetime64))


def hovmoller(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap = "Spectral_r",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    diff: bool = False,
    robust: bool = True,
    colorbar: bool | None = None,
    colorbar_label: str = "",
    xlabel: str = "time",
    ylabel: str = "latitude",
    xunits: str | None = None,
    yunits: str | None = "°N",
    title: str = "",
    width: float | str = "full",
    aspect: float = 0.5,
) -> tuple[Axes, QuadMesh]:
    """Draw a field over time (x axis) and one spatial coordinate (y axis).

    Args:
        x: Time coordinate of length ``nt``: numbers or ``datetime`` objects
            (dates get a compact date axis).
        y: Spatial coordinate of length ``ny``, e.g. latitude.
        z: Field of shape ``(nt, ny)``.
        ax: Axes to draw on; a new figure is created when ``None``.
        cmap: Colormap name or object.
        vmin: Lower colour limit; derived from ``z`` if ``None``.
        vmax: Upper colour limit; derived from ``z`` if ``None``.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        diff: Treat ``z`` as a difference: symmetric limits, and ``RdBu_r`` when
            ``cmap`` is left at its default.
        robust: Derive limits from the 2nd/98th percentiles instead of min/max.
        colorbar: Draw a colorbar. Defaults to ``True`` when this call creates the
            figure and ``False`` when drawing onto a supplied ``ax``.
        colorbar_label: Colorbar label.
        xlabel: x-axis label; ``""`` leaves the axis unlabelled.
        ylabel: y-axis label; ``""`` leaves the axis unlabelled.
        xunits: Units appended to the x-axis label.
        yunits: Units appended to the y-axis label.
        title: Axes title.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name,
            used only when a new figure is created.
        aspect: Figure ``height / width``, used only when a new figure is created.

    Returns:
        ``(ax, mesh)``: the axes and the ``QuadMesh``, for composing colorbars.
    """
    th = get_theme()
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if diff and cmap == "Spectral_r":
        cmap = "RdBu_r"
    if norm is None and (vmin is None or vmax is None):
        if diff:
            lo, hi = symmetric_limits(z)
        elif robust:
            lo, hi = robust_limits(z)
        else:
            lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    owns_fig = ax is None
    if colorbar is None:
        colorbar = owns_fig
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=aspect)

    xv = np.asarray(x)
    is_dates = _is_dates(x)
    if is_dates:
        import matplotlib.dates as mdates

        xv = mdates.date2num(list(x))

    kw = dict(cmap=resolve_cmap(cmap), shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    mesh = ax.pcolormesh(xv, y, z.T, **kw)

    if is_dates:
        from .timeseries import _format_date_axis

        _format_date_axis(ax, list(x))
    if xlabel:
        ax.set_xlabel(axis_label(xlabel, xunits))
    if ylabel:
        ax.set_ylabel(axis_label(ylabel, yunits))
    if title:
        ax.set_title(title, fontsize=th.fs_title, loc="left", pad=3)
    if colorbar:
        from .colorbars import colorbar as _cbar

        _cbar(mesh, ax, label=colorbar_label)
    return ax, mesh


def hovmoller_panels(
    fields: Mapping[str, ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    *,
    diff_fields: Collection[str] = (),
    value_cmap: str | Colormap = "Spectral_r",
    diff_cmap: str | Colormap = "RdBu_r",
    value_label: str = "",
    diff_label: str = "",
    xlabel: str = "time",
    xunits: str | None = None,
    ylabel: str = "latitude",
    yunits: str | None = "°N",
    width: float | str = "full",
    panel_aspect: float = 0.62,
    title: str = "",
    titles: Sequence[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Lay out Hovmöller panels in a row with shared value and difference colorbars.

    Value fields share robust colour limits and one colorbar; difference fields
    share symmetric limits and a second, diverging colorbar.

    Args:
        x: Shared time coordinate of length ``nt``.
        fields: ``{name: (nt, ny) array}`` in display order.
        y: Shared spatial coordinate of length ``ny``.
        diff_fields: Names in ``fields`` drawn on the difference scale.
        value_cmap: Colormap of the value panels.
        diff_cmap: Colormap of the difference panel.
        value_label: Label of the value colorbar.
        diff_label: Label of the difference colorbar.
        xlabel: Time-axis label.
        xunits: Time-axis units, appended to the label.
        ylabel: Spatial-axis label (first panel only).
        yunits: Spatial-axis units, appended to the label (first panel only).
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name.
        panel_aspect: Height / width of each panel.
        title: Figure title above the row.
        titles: Per-panel titles aligned with ``fields``; defaults to the field
            names.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat array, one per field.
    """
    from .colorbars import colorbar
    from .figure import subplots

    names = list(fields)
    diff_set = set(diff_fields)
    n = len(names)
    fig, axes = subplots(1, n, width=width, panel_aspect=panel_aspect, wspace=0.12, squeeze=False)
    axes = axes.ravel()

    val_names = [nm for nm in names if nm not in diff_set]
    dif_names = [nm for nm in names if nm in diff_set]
    val_lim = (
        robust_limits(np.concatenate([np.asarray(fields[nm]).ravel() for nm in val_names]))
        if val_names
        else None
    )
    dif_lim = (
        symmetric_limits(np.concatenate([np.asarray(fields[nm]).ravel() for nm in dif_names]))
        if dif_names
        else None
    )

    val_m = dif_m = None
    for i, (ax, nm) in enumerate(zip(axes, names)):
        is_diff = nm in diff_set
        lim = dif_lim if is_diff else val_lim
        panel_title = titles[i] if titles is not None and i < len(titles) else nm
        _, m = hovmoller(
            x,
            y,
            fields[nm],
            ax=ax,
            cmap=(diff_cmap if is_diff else value_cmap),
            vmin=lim[0],
            vmax=lim[1],
            diff=is_diff,
            colorbar=False,
            xlabel=xlabel,
            xunits=xunits,
            ylabel=(ylabel if ax is axes[0] else ""),
            yunits=yunits,
            title=panel_title,
        )
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if is_diff:
            dif_m = m
        else:
            val_m = m
    if val_m is not None:
        colorbar(
            val_m,
            [ax for ax, nm in zip(axes, names) if nm not in diff_set],
            label=value_label,
            location="bottom",
            thickness=0.03,
            shrink=0.85,
            pad=0.26,
        )
    if dif_m is not None:
        colorbar(
            dif_m,
            [ax for ax, nm in zip(axes, names) if nm in diff_set],
            label=diff_label,
            location="bottom",
            thickness=0.03,
            shrink=0.85,
            pad=0.26,
        )
    if title:
        # Sit clear above the per-panel titles (the row is short, so 1.02 collides).
        fig.suptitle(title, fontsize=get_theme().fs_panel_label, y=1.16)
    return fig, axes
