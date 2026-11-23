"""Text measurement, header drawing and colourbar placement for :func:`nicefigs.scorecard`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import text_to_path
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox, ScaledTranslation

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap, Normalize

INK = "#222222"
MUTED = "#666666"  # unit lines, 0.4 grey
NA_TEXT = "#8c8c8c"  # "n/a" and unresolved values, 0.55 grey
GREY_CELL = "#e8e8e8"

# Base font sizes (pt) of a card drawn at its printed size.
FS_VALUE = 7.5
FS_ROW = 8.0
FS_HEAD = 7.0
FS_UNIT = 6.0
FS_GROUP = 9.5
FS_CBAR = 8.0
FS_CBAR_TICK = 7.0

_LINE = 1.25  # line height as a multiple of the font size


def text_width_pt(text: str, size: float) -> float:
    """Width of a single line of ``text`` in points at font size ``size``."""
    if not text:
        return 0.0
    w, _, _ = text_to_path.get_text_width_height_descent(text, FontProperties(size=size), False)
    return float(w)


def wrap_heading(label: str, size: float, width_pt: float) -> str:
    """Break ``label`` at spaces so each line fits ``width_pt``; existing newlines are kept."""
    lines = []
    for part in str(label).split("\n"):
        line = ""
        for word in part.split():
            trial = f"{line} {word}".strip()
            if line and text_width_pt(trial, size) > width_pt:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def fit_labels(
    labels: Sequence[str],
    spans: Sequence[tuple[float, float]],
    cell_pt: float,
    size: float,
    floor: float,
    clearance_pt: float = 2.0,
) -> tuple[float, list[str]]:
    """The largest size, from ``size`` down to ``floor``, at which neighbouring labels clear.

    Each label is centred on its span (``(start, end)`` in cells) and wrapped to the span's width
    first; a word wider than the span cannot wrap, so the size shrinks in 0.1 pt steps until every
    pair of neighbouring labels is ``clearance_pt`` apart. Empty labels take no room.

    Returns:
        The size in points and the wrapped labels at that size (at ``floor`` if none clears).
    """
    centres = np.array([0.5 * (a + b) for a, b in spans], dtype=float)
    widths = [b - a for a, b in spans]
    steps = max(int(round((size - floor) / 0.1)), 0)
    for k in range(steps + 1):
        f = size - 0.1 * k
        wrapped = [wrap_heading(lab, f, w * cell_pt * 1.04) for lab, w in zip(labels, widths)]
        half = [0.5 * max(text_width_pt(line, f) for line in h.split("\n")) for h in wrapped]
        pitch = np.diff(centres) * cell_pt
        if all(p - half[j] - half[j + 1] >= clearance_pt for j, p in enumerate(pitch)):
            return f, wrapped
    return floor, wrapped


def fit_headings(
    labels: Sequence[str],
    lefts: np.ndarray,
    cell_pt: float,
    size: float,
    floor: float,
    clearance_pt: float = 2.0,
) -> tuple[float, list[str]]:
    """:func:`fit_labels` for one label per cell starting at ``lefts``."""
    return fit_labels(labels, [(x, x + 1.0) for x in lefts], cell_pt, size, floor, clearance_pt)


def group_blocks(groups: Sequence[str], lefts: np.ndarray) -> list[tuple[str, float, float]]:
    """``(name, start, end)`` of each run of equal consecutive group names, in cells."""
    blocks = []
    j = 0
    while j < len(groups):
        k = j
        while k + 1 < len(groups) and groups[k + 1] == groups[j]:
            k += 1
        blocks.append((groups[j], float(lefts[j]), float(lefts[k] + 1)))
        j = k + 1
    return blocks


def header_heights(
    headings: Sequence[str],
    unit_lines: int,
    has_groups: bool,
    scale: float,
    head_size: float | None = None,
    *,
    unit_size: float | None = None,
    group_lines: int = 1,
    group_size: float | None = None,
) -> dict[str, float]:
    """Vertical positions (pt above the top of the grid) of the stacked header elements.

    ``unit_lines`` is the most lines any column's unit spans (0 without units), so a unit with a
    second line, such as an uncertainty, keeps clear of the heading above it. ``head_size``,
    ``unit_size`` and ``group_size`` are the sizes in points of headings, units and group titles
    (default ``FS_HEAD``, ``FS_UNIT`` and ``FS_GROUP`` times ``scale``); ``group_lines`` is the most
    lines a group title spans.

    Returns:
        ``units`` and ``heading`` (bottoms of those texts), ``rule`` and ``group`` (the group rule
        and the bottom of the group titles) and ``top`` (upper edge of the whole header).
    """
    n_lines = max((h.count("\n") + 1 for h in headings), default=1)
    head_size = FS_HEAD * scale if head_size is None else head_size
    unit_size = FS_UNIT * scale if unit_size is None else unit_size
    group_size = FS_GROUP * scale if group_size is None else group_size
    units = 3.0 * scale
    heading = units + (unit_lines * unit_size * _LINE + 1.0 * scale if unit_lines else 0.0)
    top = heading + n_lines * head_size * _LINE * 0.95 + 1.0 * scale
    rule = group = top
    if has_groups:
        rule = top + 3.0 * scale
        group = rule + 2.5 * scale
        top = group + group_lines * group_size * _LINE
    return {"units": units, "heading": heading, "rule": rule, "group": group, "top": top}


def draw_header(
    ax: Axes,
    lefts: np.ndarray,
    headings: Sequence[str],
    units: Sequence[str],
    groups: Sequence[str] | None,
    heights: dict[str, float],
    *,
    group_rule: bool,
    cell_pt: float,
    scale: float,
    head_size: float | None = None,
    unit_size: float | None = None,
    group_titles: Sequence[str] | None = None,
    group_size: float | None = None,
) -> None:
    """Draw column headings, units, group titles and rules.

    Sizes default to ``FS_HEAD``, ``FS_UNIT`` and ``FS_GROUP`` times ``scale``; ``group_titles``
    holds one (wrapped) title per run of equal ``groups``, the names themselves by default.
    """
    head_size = FS_HEAD * scale if head_size is None else head_size
    unit_size = FS_UNIT * scale if unit_size is None else unit_size
    group_size = FS_GROUP * scale if group_size is None else group_size
    for x, head, unit in zip(lefts, headings, units):
        if unit:
            ax.annotate(
                unit,
                (x + 0.5, 0),
                xytext=(0, heights["units"]),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=unit_size,
                color=MUTED,
                annotation_clip=False,
            )
        ax.annotate(
            head,
            (x + 0.5, 0),
            xytext=(0, heights["heading"]),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=head_size,
            color=INK,
            linespacing=0.95,
            annotation_clip=False,
        )
    if groups is None:
        return
    shift = ScaledTranslation(0, heights["rule"] / 72.0, ax.figure.dpi_scale_trans)
    blocks = group_blocks(groups, lefts)
    titles = [name for name, _, _ in blocks] if group_titles is None else list(group_titles)
    for (_, start, end), title in zip(blocks, titles):
        centre = 0.5 * (start + end)
        title_w = max(text_width_pt(line, group_size) for line in title.split("\n"))
        half = max(0.5 * (end - start) - 0.04, 0.5 * title_w / cell_pt)
        ax.annotate(
            title,
            (centre, 0),
            xytext=(0, heights["group"]),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=group_size,
            color=INK,
            linespacing=0.95,
            annotation_clip=False,
        )
        if group_rule:
            ax.add_line(
                Line2D(
                    [centre - half, centre + half],
                    [0, 0],
                    transform=ax.transData + shift,
                    color=INK,
                    linewidth=1.6,
                    solid_capstyle="butt",
                    clip_on=False,
                )
            )


def draw_colorbar(
    ax: Axes,
    cmap: Colormap,
    norm: Normalize,
    *,
    label: str,
    extend: str,
    width_in: float,
    drop_in: float,
    thickness_in: float,
    scale: float,
) -> Colorbar:
    """Draw a slim horizontal colourbar centred under ``ax``, with ticks at band edges.

    Args:
        ax: The grid axes.
        cmap: Colormap of the bar.
        norm: Norm of the bar; a ``BoundaryNorm`` puts ticks at its band edges.
        label: Bar label.
        extend: Matplotlib ``extend`` mode.
        width_in: Bar length in inches.
        drop_in: Distance from the bottom of the grid to the bar, in inches.
        thickness_in: Bar thickness in inches.
        scale: Font-size factor.

    Returns:
        The colourbar.
    """
    from matplotlib.cm import ScalarMappable

    fig = ax.figure
    cax = fig.add_axes([0, 0, 1, 1])

    def _place(_ax, renderer):
        box = ax.get_window_extent(renderer)
        w, h, d = (v * fig.dpi for v in (width_in, thickness_in, drop_in))
        x0 = 0.5 * (box.x0 + box.x1) - 0.5 * w
        y0 = box.y0 - d - h
        return Bbox.from_bounds(x0, y0, w, h).transformed(fig.transFigure.inverted())

    cax.set_axes_locator(_place)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    kwargs = {}
    if isinstance(norm, mcolors.BoundaryNorm):
        kwargs = {"ticks": norm.boundaries, "spacing": "uniform", "format": FuncFormatter(_tick_text)}
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend, extendfrac=0.05, **kwargs)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=FS_CBAR * scale, color=INK)
    cb.ax.tick_params(length=2, width=0.5, labelsize=FS_CBAR_TICK * scale, color=INK, labelcolor=INK)
    return cb


def _tick_text(value: float, _pos=None) -> str:
    """Band-edge tick label with a true minus sign."""
    return f"{value:g}".replace("-", "−")
