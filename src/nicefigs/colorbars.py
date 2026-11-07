"""Slim colorbars flush to one or several axes, and histogram colorbars that show the data distribution."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from numpy.typing import ArrayLike

from .palettes import cmap as _resolve_cmap
from .style import get_theme


def _bbox_union(axes: Sequence[Axes]) -> tuple[float, float, float, float]:
    """Union of the display positions of ``axes`` in figure fraction coords."""
    fig = axes[0].figure
    fig.canvas.draw()
    poss = [ax.get_position() for ax in axes]
    x0 = min(p.x0 for p in poss)
    y0 = min(p.y0 for p in poss)
    x1 = max(p.x1 for p in poss)
    y1 = max(p.y1 for p in poss)
    return x0, y0, x1, y1


def colorbar(
    mappable: ScalarMappable,
    ax: Axes | Sequence[Axes],
    *,
    label: str = "",
    location: str = "right",
    pad: float = 0.012,
    thickness: float = 0.012,
    shrink: float = 1.0,
    extend: str = "neither",
    ticks: ArrayLike | None = None,
) -> Colorbar:
    """Draw a slim colorbar flush to one axes, or spanning several.

    Args:
        mappable: E.g. the return of ``imshow``, ``scatter`` or ``pcolormesh``,
            or one from :func:`make_mappable`.
        ax: One axes, or a sequence of axes the bar should span.
        label: Colorbar label.
        location: ``"right"``, ``"left"``, ``"bottom"`` or ``"top"``.
        pad: Gap between plot and bar, in figure fraction.
        thickness: Bar thickness, in figure fraction.
        shrink: Fraction of the spanned length the bar occupies (centred).
        extend: ``"neither"``, ``"both"``, ``"min"`` or ``"max"``.
        ticks: Explicit tick locations.

    Returns:
        The colorbar.
    """
    axes = list(ax) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
    fig = axes[0].figure
    th = get_theme()
    x0, y0, x1, y1 = _bbox_union(axes)

    if location in ("right", "left"):
        length = (y1 - y0) * shrink
        cy = 0.5 * (y0 + y1) - length / 2
        if location == "right":
            cax = fig.add_axes([x1 + pad, cy, thickness, length])
        else:
            cax = fig.add_axes([x0 - pad - thickness, cy, thickness, length])
        orientation = "vertical"
    else:
        length = (x1 - x0) * shrink
        cx = 0.5 * (x0 + x1) - length / 2
        if location == "bottom":
            cax = fig.add_axes([cx, y0 - pad - thickness, length, thickness])
        else:
            cax = fig.add_axes([cx, y1 + pad, length, thickness])
        orientation = "horizontal"

    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, extend=extend, ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=th.fs_tick, length=1.6, width=0.35, pad=1.2)
    if location == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    if location == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    if label:
        cb.set_label(label, fontsize=th.fs_annot, labelpad=2.0)
    return cb


def make_mappable(
    values: ArrayLike | None = None,
    *,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
) -> ScalarMappable:
    """Build a ``ScalarMappable`` for a colorbar without a plotted artist.

    Args:
        values: Data whose finite min and max fill in missing ``vmin``/``vmax``.
        cmap: Colormap or name, resolved by :func:`nicefigs.cmap`.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Normalisation; overrides ``vmin``, ``vmax`` and ``values``.
    """
    if norm is None:
        if (vmin is None or vmax is None) and values is not None:
            a = np.asarray(values, dtype=float)
            a = a[np.isfinite(a)]
            vmin = float(np.min(a)) if vmin is None else vmin
            vmax = float(np.max(a)) if vmax is None else vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=_resolve_cmap(cmap))
    sm.set_array([])
    return sm


def inset_histogram_colorbar(
    ax: Axes,
    values: ArrayLike,
    *,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    bounds: tuple[float, float, float, float] = (0.045, 0.08, 0.24, 0.17),
    n_bins: int = 24,
    label: str | None = None,
    frame_alpha: float = 0.0,
    log: bool = False,
) -> Axes | None:
    """Draw an inset histogram of ``values`` with bars coloured by ``cmap``.

    The inset acts as a colorbar that also shows the data distribution.

    Args:
        ax: Host axes.
        values: Mapped data; non-finite values are ignored.
        cmap: Colormap or name, resolved by :func:`nicefigs.cmap`.
        vmin: Histogram and colour lower limit; defaults to the 2nd percentile.
        vmax: Histogram and colour upper limit; defaults to the 98th percentile.
        norm: Colour normalisation used instead of the linear ``vmin``/``vmax`` one.
        bounds: Inset ``(x, y, width, height)`` in axes fraction; the default
            sits in the lower left, the empty South Pacific of a world map.
        n_bins: Number of bins.
        label: X-axis label of the inset.
        frame_alpha: Opacity of the inset's white background.
        log: Show ``log1p`` counts.

    Returns:
        The inset axes, or ``None`` when ``values`` has no finite entries.
    """
    th = get_theme()
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    if vmin is None or vmax is None:
        lo, hi = np.percentile(vals, [2.0, 98.0])
        vmin = float(lo) if vmin is None else vmin
        vmax = float(hi) if vmax is None else vmax
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    inset = ax.inset_axes(bounds)
    counts, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    cm = _resolve_cmap(cmap)
    if norm is not None:
        colors = [cm(float(np.clip(norm(c), 0, 1))) for c in centers]
    else:
        colors = [cm(float((c - vmin) / (vmax - vmin))) for c in centers]
    if log:
        counts = np.log1p(counts)
    inset.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=colors, edgecolor="none")
    inset.set_xlim(vmin, vmax)
    inset.set_yticks([])
    inset.xaxis.set_major_locator(mticker.MaxNLocator(3))
    inset.tick_params(axis="x", labelsize=th.fs_annot - 0.5, length=1.6, pad=0.8)
    for side in ("top", "right", "left"):
        inset.spines[side].set_visible(False)
    inset.spines["bottom"].set_linewidth(0.4)
    inset.set_facecolor("#ffffff")
    inset.patch.set_alpha(frame_alpha)
    if label is not None:
        # Left-aligned so a long label stays inside a curved map frame.
        inset.set_xlabel(label, fontsize=th.fs_annot - 0.5, labelpad=0.8, ha="left", x=0.0)
    return inset
