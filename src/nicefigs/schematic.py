"""Primitives for concept figures: schematics drawn in inches on one axes.

A concept figure (a pipeline, a data layout, a tree of merges) is built at its printed size, so a
7 pt label is 7 pt on the page. :func:`schematic_axes` makes the canvas, with the x and y data
coordinates in inches; :func:`~nicefigs.diagram_box` and :func:`~nicefigs.diagram_arrow` draw
boxes and connectors on it, and this module adds the remaining pieces: :func:`numbered_disc`,
:func:`group_band`, :func:`label_pair`, :func:`grid_cells`, :func:`folder_icon`,
:func:`cylinder_icon`, :func:`chip_icon`, and the colour
helpers :func:`tint` and :func:`shade`. Every colour defaults to the active theme and the
:data:`~nicefigs.NICE_COLORS`: ink is the theme's text colour, secondary text the neutral grey,
the accent the orange of the house palette.

Example::

    import nicefigs as nf

    nf.set_theme(nf.Theme.from_base(7.2))
    fig, ax = nf.schematic_axes(3.2, 1.0)
    nf.diagram_box(ax, (0.7, 0.5), 1.0, 0.5, text="read")
    nf.diagram_arrow(ax, (1.25, 0.5), (1.95, 0.5), style="line", width=0.7)
    nf.numbered_disc(ax, (0.25, 0.8), 1)
    nf.check_text_overlaps(fig)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from .artists import NICE_COLORS
from .figure import figure
from .style import COLORS, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

__all__ = [
    "chip_icon",
    "cylinder_icon",
    "folder_icon",
    "grid_cells",
    "group_band",
    "label_pair",
    "numbered_disc",
    "schematic_axes",
    "shade",
    "soft_color",
    "tint",
]


def tint(color: Any, amount: float) -> tuple[float, float, float]:
    """Blend ``color`` with white.

    Args:
        color: Any matplotlib colour.
        amount: Share of ``color`` that stays; 0 is white and 1 is ``color``.

    Returns:
        An RGB tuple.
    """
    rgb = np.asarray(to_rgb(color))
    r, g, b = 1 - amount * (1 - rgb)
    return float(r), float(g), float(b)


def shade(color: Any, amount: float = 0.35) -> tuple[float, float, float]:
    """Blend ``color`` with black, for text that sits next to an object of that colour.

    Args:
        color: Any matplotlib colour.
        amount: Share of black; 0 is ``color`` and 1 is black.

    Returns:
        An RGB tuple.
    """
    rgb = np.asarray(to_rgb(color))
    r, g, b = rgb * (1 - amount)
    return float(r), float(g), float(b)


def soft_color() -> str:
    """Colour of secondary text and quiet lines: the neutral grey of the house style."""
    return COLORS["neutral"]


def schematic_axes(width: float, height: float, *, theme: Any = None, **kwargs: Any) -> tuple[Figure, Axes]:
    """Create a figure of the printed size with one axes whose data units are inches.

    Args:
        width: Figure width in inches.
        height: Figure height in inches.
        theme: Theme name or :class:`~nicefigs.Theme`; ``None`` uses the active theme.
        **kwargs: Forwarded to :func:`nicefigs.figure`.

    Returns:
        The figure and its axes, which has no frame and spans the figure: x runs 0 to ``width``
        and y 0 to ``height``.
    """
    fig = figure(width=width, height=height, theme=theme, **kwargs)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def numbered_disc(
    ax: Axes,
    xy: tuple[float, float],
    n: int | str,
    *,
    radius: float = 0.07,
    color: Any = None,
    zorder: float = 12.0,
) -> Circle:
    """Draw a filled disc with a white digit, to tie a figure to the steps of a text.

    Args:
        ax: Axes to draw on.
        xy: Centre, in data coordinates.
        n: The number or letter shown.
        radius: Disc radius, in data units.
        color: Fill colour; the orange of the house palette by default.
        zorder: Drawing order; the digit draws above.

    Returns:
        The disc patch.
    """
    th = get_theme()
    disc = Circle(
        xy,
        radius,
        facecolor=NICE_COLORS["orange"] if color is None else color,
        edgecolor="white",
        linewidth=0.5,
        zorder=zorder,
    )
    ax.add_patch(disc)
    ax.text(
        xy[0],
        xy[1] - 0.004,
        str(n),
        ha="center",
        va="center",
        color="white",
        fontsize=th.fs_annot,
        fontweight="bold",
        zorder=zorder + 0.1,
    )
    return disc


def group_band(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str = "",
    *,
    color: Any = None,
    rounding: float = 0.04,
    zorder: float = 0.5,
) -> FancyBboxPatch:
    """Draw a quiet filled band that groups objects, with its title at the top left.

    Args:
        ax: Axes to draw on.
        xy: Lower-left corner, in data coordinates.
        width: Band width, in data units.
        height: Band height, in data units.
        title: Title in secondary text; empty for none.
        color: Fill colour; a light tint of the theme's text colour by default.
        rounding: Corner radius, in data units.
        zorder: Drawing order; keep it below the grouped objects.

    Returns:
        The band patch.
    """
    th = get_theme()
    band = FancyBboxPatch(
        (xy[0], xy[1]),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=tint(th.text_color, 0.06) if color is None else color,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(band)
    if title:
        ax.text(
            xy[0] + 0.08,
            xy[1] + height - 0.1,
            title,
            ha="left",
            va="center",
            color=soft_color(),
            fontsize=th.fs_annot,
            zorder=zorder + 1,
        )
    return band


def label_pair(
    ax: Axes,
    xy: tuple[float, float],
    name: str,
    detail: str,
    *,
    color: Any = None,
    ha: str = "center",
    gap: float = 0.055,
    bold: bool = False,
    zorder: float = 6.0,
) -> tuple[Text, Text]:
    """Write a name in ink with a detail in secondary text underneath, centred on ``xy``.

    Args:
        ax: Axes to draw on.
        xy: Anchor of the pair, in data coordinates.
        name: First line, in ``color``.
        detail: Second line, in the neutral grey and the smaller annotation size.
        color: Colour of the name; the theme's text colour by default.
        ha: Horizontal alignment of both lines.
        gap: Distance of each line from ``xy``, in data units.
        bold: Set the name in bold.
        zorder: Drawing order.

    Returns:
        The two text artists, name first.
    """
    th = get_theme()
    top = ax.text(
        xy[0],
        xy[1] + gap,
        name,
        ha=ha,
        va="center",
        color=th.text_color if color is None else color,
        fontsize=th.fs_legend,
        fontweight="bold" if bold else "normal",
        zorder=zorder,
    )
    bottom = ax.text(
        xy[0],
        xy[1] - gap,
        detail,
        ha=ha,
        va="center",
        color=soft_color(),
        fontsize=th.fs_annot,
        zorder=zorder,
    )
    return top, bottom


def grid_cells(
    ax: Axes,
    xy: tuple[float, float],
    nx: int,
    ny: int,
    cell: float,
    *,
    fills: Mapping[tuple[int, int], Any] | None = None,
    facecolor: Any = None,
    edgecolor: Any = None,
    linewidth: float = 0.4,
    zorder: float = 2.0,
) -> list[Rectangle]:
    """Draw a grid of ``nx`` by ``ny`` square cells, e.g. a model grid.

    Args:
        ax: Axes to draw on.
        xy: Lower-left corner of the grid, in data coordinates.
        nx: Cells along x.
        ny: Cells along y.
        cell: Cell edge, in data units.
        fills: Face colour of single cells, keyed by ``(i, j)`` with ``i`` along x and ``j``
            along y from the bottom.
        facecolor: Face colour of the other cells; a light tint of the blue of the house palette
            by default.
        edgecolor: Cell edge colour; a lighter tint of the same blue by default.
        linewidth: Edge width (pt).
        zorder: Drawing order.

    Returns:
        The cell patches, column by column.
    """
    base = NICE_COLORS["blue"]
    fc = tint(base, 0.12) if facecolor is None else facecolor
    ec = tint(base, 0.4) if edgecolor is None else edgecolor
    fills = fills or {}
    cells = []
    for i in range(nx):
        for j in range(ny):
            r = Rectangle(
                (xy[0] + i * cell, xy[1] + j * cell),
                cell,
                cell,
                facecolor=fills.get((i, j), fc),
                edgecolor=ec,
                linewidth=linewidth,
                zorder=zorder,
            )
            ax.add_patch(r)
            cells.append(r)
    return cells


def folder_icon(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: Any = None,
    edgecolor: Any = None,
    linewidth: float = 0.6,
    zorder: float = 3.0,
) -> Polygon:
    """Draw a folder icon, to stand for a directory of stored results.

    Args:
        ax: Axes to draw on.
        xy: Lower-left corner, in data coordinates.
        width: Icon width, in data units.
        height: Icon height, in data units.
        facecolor: Fill colour; a light tint of the theme's text colour by default.
        edgecolor: Outline colour; the theme's text colour by default.
        linewidth: Outline width (pt).
        zorder: Drawing order.

    Returns:
        The icon patch.
    """
    th = get_theme()
    x, y = xy
    tab = height * 0.22
    pts = [
        (x, y),
        (x, y + height),
        (x + width * 0.4, y + height),
        (x + width * 0.5, y + height - tab),
        (x + width, y + height - tab),
        (x + width, y),
    ]
    icon = Polygon(
        pts,
        closed=True,
        facecolor=tint(th.text_color, 0.12) if facecolor is None else facecolor,
        edgecolor=th.text_color if edgecolor is None else edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(icon)
    return icon


def cylinder_icon(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: Any = None,
    edgecolor: Any = None,
    linewidth: float = 0.6,
    zorder: float = 3.0,
) -> tuple[Polygon, Polygon]:
    """Draw a cylinder icon, to stand for a data store or database.

    Args:
        ax: Axes to draw on.
        xy: Lower-left corner of the icon's bounding box, in data coordinates.
        width: Icon width, in data units.
        height: Icon height, in data units, rim included.
        facecolor: Fill colour; a light tint of the theme's text colour by default.
        edgecolor: Outline colour; the theme's text colour by default.
        linewidth: Outline width (pt).
        zorder: Drawing order; the top rim draws just above the body.

    Returns:
        The body and the top rim patches.
    """
    th = get_theme()
    face = tint(th.text_color, 0.12) if facecolor is None else facecolor
    edge = th.text_color if edgecolor is None else edgecolor
    x, y = xy
    a, b = width / 2, min(0.14 * width, 0.2 * height)
    cx = x + a
    t = np.linspace(0.0, np.pi, 41)
    lower = np.column_stack([cx - a * np.cos(t), y + b - b * np.sin(t)])
    upper = np.column_stack([cx + a * np.cos(t), y + height - b + b * np.sin(t)])
    body = Polygon(
        np.vstack([lower, upper]),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    s = np.linspace(0.0, 2 * np.pi, 81)
    rim = Polygon(
        np.column_stack([cx + a * np.cos(s), y + height - b + b * np.sin(s)]),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder + 0.01,
    )
    ax.add_patch(body)
    ax.add_patch(rim)
    return body, rim


def chip_icon(
    ax: Axes,
    center: tuple[float, float],
    size: float,
    *,
    pins: int = 3,
    facecolor: Any = None,
    edgecolor: Any = None,
    linewidth: float = 0.6,
    zorder: float = 3.0,
) -> Polygon:
    """Draw a processor chip icon, a square die with pins on every side, to stand for a CPU or GPU.

    Args:
        ax: Axes to draw on.
        center: Centre, in data coordinates.
        size: Side of the icon with its pins, in data units; the die is 70 % of it.
        pins: Pins per side.
        facecolor: Fill colour of the die; a light tint of the theme's text colour by default.
        edgecolor: Outline and pin colour; the theme's text colour by default.
        linewidth: Outline and pin width (pt).
        zorder: Drawing order.

    Returns:
        The die patch.
    """
    th = get_theme()
    edge = th.text_color if edgecolor is None else edgecolor
    cx, cy = center
    h, d = size / 2, 0.35 * size
    die = Polygon(
        [(cx - d, cy - d), (cx + d, cy - d), (cx + d, cy + d), (cx - d, cy + d)],
        closed=True,
        facecolor=tint(th.text_color, 0.12) if facecolor is None else facecolor,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(die)
    for k in range(pins):
        u = -d + (k + 1) * 2 * d / (pins + 1)
        for xs, ys in (
            ((cx + u, cx + u), (cy + d, cy + h)),
            ((cx + u, cx + u), (cy - d, cy - h)),
            ((cx - d, cx - h), (cy + u, cy + u)),
            ((cx + d, cx + h), (cy + u, cy + u)),
        ):
            ax.plot(xs, ys, color=edge, linewidth=linewidth, solid_capstyle="butt", zorder=zorder - 0.01)
    return die
