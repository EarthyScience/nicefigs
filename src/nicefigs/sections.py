"""Vertical cross-sections (curtains) and vertical profiles."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .figure import subplots
from .palettes import cmap as resolve_cmap
from .palettes import qualitative, robust_limits, symmetric_limits
from .style import get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


def profile(
    profiles: Mapping[str, ArrayLike] | ArrayLike,
    y: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    invert_y: bool = True,
    log_y: bool = False,
    colors: Mapping[str, str] | Sequence[str] | None = None,
    styles: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    marker: str = "o",
    baselines: Mapping[str, ArrayLike] | None = None,
    baseline_colors: Mapping[str, str] | None = None,
    xlabel: str = "",
    xunits: str | None = None,
    ylabel: str = "level",
    yunits: str | None = None,
    title: str = "",
    legend: bool = True,
    width: float | str = "half",
    aspect: float = 0.95,
) -> Axes:
    """Draw one or more vertical profiles: value on x, level on y.

    The 1-D companion to :func:`cross_section`. The vertical axis is inverted by
    default so that descending pressure levels put the surface at the bottom.

    Args:
        profiles: ``{name: values}`` along ``y``, or a single array.
        y: Vertical coordinate shared by all profiles (e.g. pressure); defaults
            to the level index.
        ax: Axes to draw on; a new figure is created when ``None``.
        invert_y: Invert the vertical axis.
        log_y: Log-scale the vertical axis.
        colors: ``{name: colour}`` or a list cycled over the profiles; defaults to
            the theme's qualitative palette.
        styles: ``{name: linestyle}``; default solid.
        markers: ``{name: marker}`` overriding ``marker``.
        marker: Marker used for profiles not in ``markers``.
        baselines: ``{name: values}`` reference profiles drawn dashed.
        baseline_colors: ``{name: colour}`` for ``baselines``; default grey.
        xlabel: Value-axis label.
        xunits: Value-axis units, appended to the label.
        ylabel: Vertical-axis label.
        yunits: Vertical-axis units, appended to the label.
        title: Axes title.
        legend: Draw a legend when there is more than one line.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name,
            used only when a new figure is created.
        aspect: Figure ``height / width``, used only when a new figure is created.

    Returns:
        The axes.
    """
    th = get_theme()
    if not isinstance(profiles, dict):
        profiles = {"profile": profiles}
    names = list(profiles)
    palette = colors or {nm: c for nm, c in zip(names, qualitative(len(names)))}
    if not isinstance(palette, dict):
        palette = {nm: palette[i % len(palette)] for i, nm in enumerate(names)}
    marks = markers or {}

    if ax is None:
        _, ax = subplots(width=width, aspect=aspect)

    for nm in names:
        v = np.asarray(profiles[nm], dtype=float)
        yy = np.arange(v.size) if y is None else np.asarray(y, dtype=float)
        ax.plot(
            v,
            yy,
            linestyle=(styles or {}).get(nm, "-"),
            color=palette.get(nm),
            marker=marks.get(nm, marker),
            markersize=4,
            linewidth=th.lw_line,
            label=nm,
        )
    if baselines:
        bpal = baseline_colors or {}
        for nm, v in baselines.items():
            v = np.asarray(v, dtype=float)
            yy = np.arange(v.size) if y is None else np.asarray(y, dtype=float)
            ax.plot(
                v, yy, linestyle="--", color=bpal.get(nm, "#888888"), linewidth=th.lw_line * 0.8, label=nm
            )

    if log_y:
        ax.set_yscale("log")
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()

    from .units import axis_label

    ax.set_xlabel(axis_label(xlabel, xunits))
    ax.set_ylabel(axis_label(ylabel, yunits))
    if title:
        ax.set_title(title, fontsize=th.fs_title, loc="left", pad=3)
    if legend and (len(names) > 1 or baselines):
        ax.legend(fontsize=th.fs_legend, framealpha=0.9)
    return ax


def cross_section(
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
    invert_y: bool = True,
    xlabel: str = "latitude",
    ylabel: str = "pressure",
    xunits: str | None = "°N",
    yunits: str | None = "hPa",
    title: str = "",
    log_y: bool = False,
) -> tuple[Axes, QuadMesh]:
    """Draw a field over a horizontal and a vertical coordinate as a curtain.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        x: Horizontal coordinate of length ``nx`` (e.g. latitude).
        y: Vertical coordinate of length ``ny`` (e.g. pressure).
        z: Field of shape ``(nx, ny)``.
        cmap: Colormap name or object.
        vmin: Lower colour limit; derived from ``z`` if ``None``.
        vmax: Upper colour limit; derived from ``z`` if ``None``.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        diff: Treat ``z`` as a difference: symmetric limits, and ``RdBu_r`` when
            ``cmap`` is left at its default.
        robust: Derive limits from the 2nd/98th percentiles instead of min/max.
        invert_y: Invert the vertical axis (surface at the bottom for pressure).
        xlabel: x-axis label.
        ylabel: y-axis label.
        xunits: Units appended to the x-axis label.
        yunits: Units appended to the y-axis label.
        title: Axes title.
        log_y: Log-scale the vertical axis.

    Returns:
        ``(ax, mesh)``: the axes and the drawn :class:`~matplotlib.collections.QuadMesh`.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    x = np.asarray(x, dtype=float)
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

    kw = dict(cmap=resolve_cmap(cmap), shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    mesh = ax.pcolormesh(x, y, z.T, **kw)

    if log_y:
        ax.set_yscale("log")
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()

    from .units import axis_label

    ax.set_xlabel(axis_label(xlabel, xunits))
    ax.set_ylabel(axis_label(ylabel, yunits))
    if title:
        ax.set_title(title, fontsize=th.fs_title, loc="left", pad=3)
    return ax, mesh


def cross_section_panels(
    fields: Mapping[str, ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    *,
    diff_fields: Collection[str] = (),
    value_cmap: str | Colormap = "Spectral_r",
    diff_cmap: str | Colormap = "RdBu_r",
    value_label: str = "",
    diff_label: str = "",
    invert_y: bool = True,
    log_y: bool = False,
    width: float | str = "full",
    panel_aspect: float = 0.8,
    ylabel: str = "pressure",
    yunits: str | None = "hPa",
    xlabel: str = "latitude",
    xunits: str | None = "°N",
    title: str = "",
    titles: Sequence[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Lay out curtains in a row with shared value and difference colorbars.

    Value fields share robust colour limits and one colorbar; difference fields
    share symmetric limits and a second, diverging colorbar.

    Args:
        fields: ``{name: (nx, ny) array}`` in display order.
        x: Shared horizontal coordinate of length ``nx``.
        y: Shared vertical coordinate of length ``ny``.
        diff_fields: Names in ``fields`` drawn on the difference scale.
        value_cmap: Colormap of the value panels.
        diff_cmap: Colormap of the difference panel.
        value_label: Label of the value colorbar.
        diff_label: Label of the difference colorbar.
        invert_y: Invert the vertical axis.
        log_y: Log-scale the vertical axis.
        width: Figure width in inches or a :data:`nicefigs.figure.WIDTHS` name.
        panel_aspect: Height / width of each panel.
        ylabel: Vertical-axis label (first panel only).
        yunits: Vertical-axis units, appended to the label (first panel only).
        xlabel: Horizontal-axis label.
        xunits: Horizontal-axis units, appended to the label.
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
    fig, axes = subplots(1, n, width=width, panel_aspect=panel_aspect, wspace=0.14, squeeze=False)
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
        _, m = cross_section(
            x,
            y,
            fields[nm],
            cmap=(diff_cmap if is_diff else value_cmap),
            vmin=lim[0],
            vmax=lim[1],
            diff=is_diff,
            invert_y=invert_y,
            log_y=log_y,
            xlabel=xlabel,
            xunits=xunits,
            ylabel=(ylabel if ax is axes[0] else ""),
            yunits=yunits,
            title=panel_title,
            ax=ax,
        )
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if is_diff:
            dif_m = m
        else:
            val_m = m
    # Extra pad so the horizontal bars sit clear below the x-axis labels.
    if val_m is not None:
        colorbar(
            val_m,
            [ax for ax, nm in zip(axes, names) if nm not in diff_set],
            label=value_label,
            location="bottom",
            thickness=0.03,
            shrink=0.85,
            pad=0.14,
        )
    if dif_m is not None:
        colorbar(
            dif_m,
            [ax for ax, nm in zip(axes, names) if nm in diff_set],
            label=diff_label,
            location="bottom",
            thickness=0.03,
            shrink=0.85,
            pad=0.14,
        )
    if title:
        # Sit clear above the per-panel titles (the row is short, so 1.02 collides).
        fig.suptitle(title, fontsize=get_theme().fs_panel_label, y=1.16)
    return fig, axes
