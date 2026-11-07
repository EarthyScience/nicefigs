"""Timeseries of named series against dates, with compact date axes and bands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .palettes import qualitative
from .scatter import scatter as _scatter
from .style import get_theme
from .units import axis_label


def _format_date_axis(ax, dates, *, maxticks=5):
    """Choose reasonable, compact date locators/formatters for the observed span.

    Uses matplotlib's ``ConciseDateFormatter`` (which factors the year/month out
    into a single offset label) so ticks stay legible even in narrow facet
    panels; very long spans fall back to bare years.

    Args:
        ax: Axes whose x-axis carries the dates.
        dates: Sequence of ``datetime.datetime``/``datetime.date`` or
            ``numpy.datetime64`` values.
        maxticks: Upper bound passed to the date locator.
    """
    arr = np.asarray(list(dates))
    span = arr.max() - arr.min()
    span_days = span / np.timedelta64(1, "D") if isinstance(span, np.timedelta64) else span.days
    loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
    if span_days > 6 * 365:
        fmt = mdates.DateFormatter("%Y")
    else:
        fmt = mdates.ConciseDateFormatter(loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)


def timeseries(
    dates: Sequence[Any] | ArrayLike,
    series: Mapping[str, ArrayLike] | ArrayLike,
    *,
    ax: Axes | None = None,
    bands: Mapping[str, Any] | None = None,
    colors: Mapping[str, Any] | None = None,
    styles: Mapping[str, str] | None = None,
    linewidth: float | None = None,
    obs: tuple[Any, ArrayLike] | Mapping[str, tuple[Any, ArrayLike]] | None = None,
    obs_label: str = "obs",
    obs_color: Any = "#222222",
    obs_size: float = 9,
    obs_marker: str = "x",
    obs_hue: ArrayLike | Mapping[str, ArrayLike] | None = None,
    obs_cmap: str = "viridis",
    obs_vmin: float | None = None,
    obs_vmax: float | None = None,
    obs_hue_label: str = "",
    obs_colorbar: bool = False,
    annotation: str = "",
    ylabel: str = "",
    yunits: str | None = None,
    xlabel: str = "",
    title: str = "",
    legend: bool = True,
    highlight: str | None = None,
    marker: str | None = None,
    width: float | str = "full",
    aspect: float = 0.4,
    band_alpha: float = 0.18,
) -> Axes:
    """Plot named series against ``dates`` on a date axis sized to the span.

    Args:
        dates: Shared ``datetime``/``date``/``datetime64`` x values.
        series: ``{name: values}``, or a single array for one unnamed series.
        ax: Axes to draw on; a new figure is created if ``None``.
        bands: ``{name: (lower, upper)}`` for one uncertainty band, or
            ``{name: [(lo1, hi1), (lo2, hi2), ...]}`` for nested bands. Nested
            bands are drawn widest first with opacity rising toward the centre,
            in any input order.
        colors: ``{name: colour}``; defaults to the theme palette.
        styles: ``{name: linestyle}``.
        linewidth: Line width (pt); defaults to the theme's.
        obs: Sparse observations drawn as markers, ``(dates, values)`` or
            ``{label: (dates, values)}``.
        obs_label: Legend label when ``obs`` is a single tuple.
        obs_color: Marker colour of ``obs`` without ``obs_hue``.
        obs_size: Marker area of ``obs`` (pt²).
        obs_marker: Marker shape of ``obs``.
        obs_hue: Continuous values colouring the ``obs`` markers on ``obs_cmap``;
            an array for a single ``obs`` tuple, or ``{label: values}`` matching
            an ``obs`` mapping.
        obs_cmap: Colormap for ``obs_hue``.
        obs_vmin: Lower ``obs_hue`` colour limit.
        obs_vmax: Upper ``obs_hue`` colour limit.
        obs_hue_label: Colorbar label for ``obs_hue``.
        obs_colorbar: Add a colorbar for ``obs_hue``.
        annotation: Text box in the upper-left corner.
        ylabel: y-axis label.
        yunits: Units appended to ``ylabel``.
        xlabel: x-axis label.
        title: Left-aligned panel title.
        legend: Add a legend when there is more than one labelled entry.
        highlight: Name of a series to draw thicker and on top.
        marker: Marker shape on the series lines.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        aspect: Figure height / width when ``ax`` is ``None``.
        band_alpha: Opacity of a single band; the innermost of nested bands.

    Returns:
        The axes.
    """
    th = get_theme()
    if not isinstance(series, Mapping):
        series = {"series": np.asarray(series)}
    names = list(series)
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=aspect)
    cols = colors or dict(zip(names, qualitative(len(names))))
    styles = styles or {}
    lw = linewidth if linewidth is not None else th.lw_line

    for name in names:
        vals = np.asarray(series[name], dtype=float)
        is_hi = name == highlight
        ax.plot(
            dates,
            vals,
            color=cols[name],
            lw=lw * (1.4 if is_hi else 1.0),
            linestyle=styles.get(name, "-"),
            zorder=(5 if is_hi else 3),
            label=name,
            marker=marker,
            markersize=3 if marker else 0,
        )
        if bands and name in bands:
            raw = bands[name]
            band_list = raw if isinstance(raw, list) else [raw]
            band_list = sorted(
                band_list,
                key=lambda b: float(np.nanmax(np.asarray(b[1], dtype=float) - np.asarray(b[0], dtype=float))),
                reverse=True,
            )
            n_bands = len(band_list)
            for bi, (lo, hi) in enumerate(band_list):
                a = band_alpha * (bi + 1) / n_bands if n_bands > 1 else band_alpha
                ax.fill_between(dates, lo, hi, color=cols[name], alpha=a, linewidth=0, zorder=1)

    proxy_handles = []
    if obs is not None:
        obs_map = obs if isinstance(obs, Mapping) else {obs_label: obs}
        hue_map = (
            {}
            if obs_hue is None
            else (obs_hue if isinstance(obs_hue, Mapping) else {next(iter(obs_map)): obs_hue})
        )
        # 'x' markers read cleanly against the model lines; thin stroke, no fill.
        lw = 0.7 if obs_marker in ("x", "+", ".", "1", "2") else 0
        hue_mappable = None
        for label, (odates, ovals) in obs_map.items():
            hue_vals = hue_map.get(label)
            if hue_vals is not None:
                # Per-point colour would give the legend a meaningless swatch, so draw
                # unlabelled and add one neutral proxy handle. scatter() casts x to
                # float, which datetimes reject, hence date2num.
                _, hue_mappable = _scatter(
                    mdates.date2num(odates),
                    np.asarray(ovals, dtype=float),
                    hue=hue_vals,
                    hue_cmap=obs_cmap,
                    hue_label=obs_hue_label,
                    vmin=obs_vmin,
                    vmax=obs_vmax,
                    s=obs_size,
                    marker=obs_marker,
                    alpha=0.8,
                    edgecolor="none",
                    linewidth=lw,
                    zorder=6,
                    legend=False,
                    ax=ax,
                )
                proxy_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=th.text_color,
                        marker=obs_marker,
                        linestyle="None",
                        markersize=np.sqrt(obs_size),
                        label=label,
                    )
                )
            else:
                ax.scatter(
                    odates,
                    np.asarray(ovals, dtype=float),
                    s=obs_size,
                    marker=obs_marker,
                    color=obs_color,
                    alpha=0.8,
                    linewidths=lw,
                    zorder=6,
                    label=label,
                )
        if obs_colorbar and hue_mappable is not None:
            from .colorbars import colorbar as _cb

            _cb(hue_mappable, ax, label=obs_hue_label)

    _format_date_axis(ax, dates)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(axis_label(ylabel, yunits) if ylabel else "")
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    if annotation:
        ax.text(
            0.02,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=th.fs_annot,
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.4, alpha=0.85),
        )
    n_lab = len(names) + (0 if obs is None else (len(obs) if isinstance(obs, Mapping) else 1))
    if legend and n_lab > 1:
        if proxy_handles:
            handles, labels = ax.get_legend_handles_labels()
            handles = handles + proxy_handles
            labels = labels + [h.get_label() for h in proxy_handles]
            ax.legend(
                handles=handles,
                labels=labels,
                loc="best",
                fontsize=th.fs_legend,
                ncol=min(n_lab, 3),
                handlelength=1.2,
                columnspacing=1.0,
            )
        else:
            ax.legend(
                loc="best", fontsize=th.fs_legend, ncol=min(n_lab, 3), handlelength=1.2, columnspacing=1.0
            )
    ax.margins(x=0.01)
    return ax
