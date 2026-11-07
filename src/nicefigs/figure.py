"""Figure creation at physical sizes, saving, and text-layout checks.

Give one dimension (usually a named width such as ``"single"``) and an
aspect ratio; the other dimension is solved. Fonts are in points, so they render
at exactly the theme's sizes::

    fig, ax = nf.subplots(width="single", aspect=0.75)
    fig, axs = nf.subplots(2, 3, width="full", panel_aspect=1.0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

from .style import Theme, get_theme, style

MM = 1.0 / 25.4  # mm → inch

#: Named target *widths* in inches. Typical journal values:
#: ``"single"`` is one column of a two-column journal (Nature 89 mm, AGU 95 mm,
#: ICML 3.25 in); ``"double"`` spans both columns (Nature 183 mm, AGU 190 mm,
#: ICML 6.75 in); ``"full"`` is the text width of a one-column article
#: (NeurIPS 5.5 in); ``"half"`` and ``"third"`` are fractions of it.
WIDTHS: dict[str, float] = {
    "single": 89 * MM,
    "onehalf": 120 * MM,
    "double": 183 * MM,
    "full": 5.5,
    "half": 2.75,
    "third": 1.83,
    # 16:9 presentation slide (13.33 in wide) and fractions of it.
    "slide": 13.33,
    "slide-half": 6.5,
    "slide-third": 4.2,
}

#: Named *full figure sizes* (w, h) in inches — for whole-slide / poster canvases.
CANVASES: dict[str, tuple[float, float]] = {
    "slide-16-9": (13.33, 7.5),
    "slide-4-3": (10.0, 7.5),
    "a4-portrait": (8.27, 11.69),
    "a4-landscape": (11.69, 8.27),
    "poster-a0-portrait": (33.1, 46.8),
    "poster-a0-landscape": (46.8, 33.1),
}

#: Default ``height / width`` of a figure when no aspect is given.
DEFAULT_ASPECT = 0.66


def figure_size(
    width: float | str | None = None,
    height: float | None = None,
    aspect: float | None = None,
    *,
    nrows: int = 1,
    ncols: int = 1,
    panel_aspect: float | None = None,
) -> tuple[float, float]:
    """Solve a ``(width, height)`` in inches from a flexible spec.

    Args:
        width: Inches, or a name from :data:`WIDTHS` / :data:`CANVASES`.
        height: Inches. Give *either* ``width`` or ``height`` (or both).
        aspect: Whole-figure ``height / width``. Ignored if both dims are given.
        nrows: Panel rows; only used with ``panel_aspect``.
        ncols: Panel columns; only used with ``panel_aspect``.
        panel_aspect: Per-panel ``height / width``. If set, the figure aspect is
            derived so each panel has this shape (``nrows/ncols * panel_aspect``).
            Takes precedence over ``aspect``.

    Returns:
        ``(width, height)`` in inches. With nothing given, a ``"half"``
        wide panel at :data:`DEFAULT_ASPECT`.

    Raises:
        KeyError: ``width`` is an unknown preset name.
    """
    if isinstance(width, str):
        if width in CANVASES and height is None and aspect is None and panel_aspect is None:
            return CANVASES[width]
        if width in CANVASES:
            width = CANVASES[width][0]
        elif width in WIDTHS:
            width = WIDTHS[width]
        else:
            raise KeyError(f"Unknown width preset {width!r}. Known: {sorted(WIDTHS) + sorted(CANVASES)}")

    if panel_aspect is not None:
        aspect = (nrows / ncols) * panel_aspect

    if width is not None and height is not None:
        return float(width), float(height)
    if width is not None:
        return float(width), float(width) * (aspect if aspect is not None else DEFAULT_ASPECT)
    if height is not None:
        asp = aspect if aspect is not None else DEFAULT_ASPECT
        return float(height) / asp, float(height)
    return WIDTHS["half"], WIDTHS["half"] * DEFAULT_ASPECT


def figure(
    width: float | str | None = None,
    height: float | None = None,
    aspect: float | None = None,
    *,
    theme: str | Theme | None = None,
    **kwargs: Any,
) -> Figure:
    """Create an empty figure at a physical size.

    Args:
        width: Inches, or a preset name; see :func:`figure_size`.
        height: Inches.
        aspect: Figure ``height / width``.
        theme: Theme applied while the figure is created; ``None`` uses the
            active theme. Use :func:`nicefigs.set_theme` to keep it afterwards.
        **kwargs: Forwarded to :func:`matplotlib.pyplot.figure`.

    Returns:
        The new figure.
    """
    size = figure_size(width, height, aspect)
    th = get_theme(theme)
    with style(th):
        fig = plt.figure(figsize=size, dpi=th.dpi, **kwargs)
    return fig


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    width: float | str | None = None,
    height: float | None = None,
    aspect: float | None = None,
    panel_aspect: float | None = None,
    theme: str | Theme | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """:func:`matplotlib.pyplot.subplots` at a physical size, with a theme applied.

    Args:
        nrows: Panel rows.
        ncols: Panel columns.
        width: Inches, or a preset name; see :func:`figure_size`.
        height: Inches.
        aspect: Whole-figure ``height / width``.
        panel_aspect: Per-panel ``height / width``; overrides ``aspect``.
        theme: Theme applied while the figure is created; ``None`` uses the
            active theme.
        **kwargs: Forwarded to :func:`matplotlib.pyplot.subplots`;
            ``wspace``, ``hspace``, ``width_ratios`` and ``height_ratios`` are
            routed into ``gridspec_kw``.

    Returns:
        ``(fig, axes)`` as returned by :func:`matplotlib.pyplot.subplots`: one
        Axes, or an array of them.
    """
    size = figure_size(width, height, aspect, nrows=nrows, ncols=ncols, panel_aspect=panel_aspect)
    gridspec_kw = dict(kwargs.pop("gridspec_kw", None) or {})
    for k in ("wspace", "hspace", "width_ratios", "height_ratios"):
        if k in kwargs:
            gridspec_kw[k] = kwargs.pop(k)
    if gridspec_kw:
        kwargs["gridspec_kw"] = gridspec_kw
    th = get_theme(theme)
    with style(th):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, dpi=th.dpi, **kwargs)
    return fig, axes


def save(
    fig: Figure,
    path: str | os.PathLike[str],
    *,
    formats: tuple[str, ...] | None = None,
    dpi: int | None = None,
    transparent: bool = False,
    close: bool = False,
) -> list[Path]:
    """Save ``fig`` with a tight bounding box, creating parent directories.

    Args:
        fig: The figure.
        path: Output file. With ``formats``, its suffix is replaced by each format.
        formats: Extensions to write, e.g. ``("pdf", "png")``.
        dpi: Raster resolution; defaults to the active theme's ``savedpi``.
        transparent: Transparent background.
        close: Close the figure after saving.

    Returns:
        The written paths, one per format, in the order of ``formats``.

    Example::

        nf.save(fig, "figures/fig01", formats=("pdf", "png"))
    """
    p = Path(path)
    th = get_theme()
    dpi = dpi or th.savedpi
    paths = []
    if formats is None:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
        paths.append(p)
    else:
        stem = p.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in formats:
            out = stem.with_suffix(f".{ext.lstrip('.')}")
            fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
            paths.append(out)
    if close:
        plt.close(fig)
    return paths


@dataclass(frozen=True)
class TextIssue:
    """A layout problem found by :func:`check_text_overlaps`.

    Attributes:
        kind: ``"overlap"`` (two text boxes intersect) or ``"clipped"`` (a box leaves the figure).
        texts: The strings involved: two for an overlap, one for a clipped text.
    """

    kind: str
    texts: tuple[str, ...]


def _tick_labels(fig: Figure) -> tuple[set[int], set[int]]:
    """Ids of all tick-label texts, and of those whose tick lies outside its axis' view (never drawn)."""
    every, hidden = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis, *([ax.zaxis] if hasattr(ax, "zaxis") else [])):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                labels = {id(tick.label1), id(tick.label2)}
                every |= labels
                loc = tick.get_loc()
                if loc is None or not lo - 1e-9 * (hi - lo) <= loc <= hi + 1e-9 * (hi - lo):
                    hidden |= labels
    return every, hidden


def check_text_overlaps(
    fig: Figure,
    *,
    ignore_ticks: bool = False,
    clipped: bool = True,
    tolerance: float = 1.0,
) -> list[TextIssue]:
    """Find text that overlaps other text or runs off the figure, as the renderer draws it.

    Every visible, non-empty text artist is measured by its rendered bounding box (titles, axis and
    tick labels, colorbar labels, legend entries, annotations). Boxes are axis-aligned, so rotated
    labels count by their enclosing rectangle.

    Args:
        fig: The figure; it is drawn once to lay out its text.
        ignore_ticks: Leave tick labels out entirely.
        clipped: Also report text whose box leaves the figure canvas. Saving with
            ``bbox_inches="tight"`` (as :func:`save` does) grows the canvas instead of cutting.
        tolerance: Pixels a box may cross the figure edge or another box before it counts.

    Returns:
        One :class:`TextIssue` per overlapping pair and per clipped text; empty when the layout is
        clean.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks, hidden = _tick_labels(fig)
    skip = ticks if ignore_ticks else hidden
    boxes = []
    for t in fig.findobj(Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():
            continue
        boxes.append((t.get_text(), t.get_window_extent(renderer).padded(-tolerance / 2)))
    issues = []
    for i, (s1, b1) in enumerate(boxes):
        for s2, b2 in boxes[i + 1 :]:
            if b1.overlaps(b2):
                issues.append(TextIssue("overlap", (s1, s2)))
    if clipped:
        fb = fig.bbox
        for s, b in boxes:
            if (
                b.x0 < fb.x0 - tolerance
                or b.y0 < fb.y0 - tolerance
                or b.x1 > fb.x1 + tolerance
                or b.y1 > fb.y1 + tolerance
            ):
                issues.append(TextIssue("clipped", (s,)))
    return issues
