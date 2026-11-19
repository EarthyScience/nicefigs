"""Box-and-arrow diagrams: flowcharts, schematics and box-model budgets.

:func:`diagram_box` and :func:`diagram_arrow` are the building blocks: a labelled, rounded
box and a filled arrow whose width is set in points, both placed in data coordinates, for
flowcharts and schematics of any layout. :func:`flux_diagram` composes them into a box
model's budget: boxes in rows (stacked layers, bottom first) and columns (side by side),
coloured by a value on a diverging scale, with every flux an arrow whose direction follows
its sign and whose width grows with the square root of its magnitude. :func:`flux_legend`
explains the encoding.

Example::

    import nicefigs as nf
    from nicefigs import data

    b = data.box_budget()
    ax, diagram = nf.flux_diagram(b["storage"], vertical=b["vertical"], ground=b["ground"])
    nf.flux_legend(diagram, {"box": "storage change", "vertical": "upward flux"})
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import ArrowStyle, FancyArrowPatch, FancyBboxPatch, Rectangle

from .palettes import cmap as resolve_cmap
from .palettes import diverging_norm, legend_handle
from .style import COLORS, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from numpy.typing import ArrayLike

__all__ = [
    "FLUX_COLORS",
    "FluxArrow",
    "FluxDiagram",
    "diagram_arrow",
    "diagram_box",
    "flux_diagram",
    "flux_legend",
]

#: Default colour of each kind of flux; ``"box"`` is not a flux and takes the colormap.
FLUX_COLORS: dict[str, str] = {
    "vertical": "#b2182b",
    "horizontal": COLORS["primary"],
    "lateral": COLORS["primary"],
    "ground": "#9a9a9a",
}

#: Kinds of flux, in the order a legend lists them.
_KINDS = ("lateral", "horizontal", "vertical", "ground")

# Geometry in data units: column ``j`` is centred on ``x = j + 0.5`` and row ``k`` on ``y = k``.
_GAP = 0.02  # between an arrow's end and the box edge
_LATERAL_SPAN = 0.16  # length of a lateral arrow
_GROUND_Y = -0.66  # the ground line, below row 0
_LABEL_DROP = 0.2  # from the ground line (or the bottom boxes) down to the column labels
_MIN_WIDTH = 0.4  # pt, the width of a zero flux


@dataclass(frozen=True)
class FluxArrow:
    """One drawn flux.

    Attributes:
        kind: ``"vertical"``, ``"horizontal"``, ``"lateral"`` or ``"ground"``.
        index: Index of the value in its input array; a lateral flux draws two arrows with
            the same index.
        value: The value drawn.
        tail: ``(x, y)`` of the arrow's tail, in data coordinates.
        head: ``(x, y)`` of its tip, in data coordinates.
        width: Shaft width (pt).
        artist: The patch drawn.
    """

    kind: str
    index: tuple[int, ...]
    value: float
    tail: tuple[float, float]
    head: tuple[float, float]
    width: float
    artist: FancyArrowPatch


@dataclass
class FluxDiagram:
    """What :func:`flux_diagram` drew, for a colorbar, a legend or tests.

    Attributes:
        ax: The axes drawn on.
        mappable: Box colormap and norm, for :func:`nicefigs.colorbar`.
        boxes: ``(n_rows, n_cols)`` object array of the box patches.
        arrows: Every flux arrow, in drawing order.
        colors: Colour of each kind of flux.
        part_colors: Colour of each part of the vertical flux.
    """

    ax: Axes
    mappable: ScalarMappable
    boxes: np.ndarray
    arrows: list[FluxArrow] = field(default_factory=list)
    colors: dict[str, str] = field(default_factory=dict)
    part_colors: dict[str, str] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list[FluxArrow]:
        """The arrows of one kind of flux."""
        return [a for a in self.arrows if a.kind == kind]


def _fmt(value: float) -> str:
    """Three significant digits, at least two decimals below 1, with a true minus sign."""
    if not np.isfinite(value) or value == 0.0:
        return "0.00"
    digits = max(0, min(2, 2 - int(np.floor(np.log10(abs(value))))))
    text = f"{value:.{digits}f}"
    return text.removeprefix("-") if float(text) == 0.0 else text.replace("-", "−")


def _as_grid(values: ArrayLike | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if values is None:
        return None
    a = np.asarray(values, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    return a


def _reference(values: np.ndarray | None, given: float | None) -> float:
    if given is not None:
        return float(given)
    if values is None or not np.isfinite(values).any():
        return 0.0
    return float(np.nanmax(np.abs(values)))


def _ink(fill: Any) -> str:
    """Dark or white text, whichever reads on ``fill``."""
    r, g, b, _ = mcolors.to_rgba(fill)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else "#1a1a1a"


def diagram_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    text: str = "",
    facecolor: Any = "white",
    edgecolor: Any = "0.55",
    linewidth: float = 0.5,
    rounding: float = 0.05,
    textcolor: Any = None,
    linestyle: Any = "-",
    hatch: str | None = None,
    zorder: float = 2.0,
    **text_kw: Any,
) -> FancyBboxPatch:
    """Draw a rounded box centred on ``xy``, optionally with centred text.

    Args:
        ax: Axes to draw on.
        xy: Centre of the box, in data coordinates.
        width: Box width, in data units.
        height: Box height, in data units.
        text: Text centred in the box; multi-line text is fine.
        facecolor: Fill colour; ``"none"`` for an outline.
        edgecolor: Outline colour.
        linewidth: Outline width (pt).
        rounding: Corner radius, in data units; 0 gives square corners.
        textcolor: Text colour; by default dark or white, whichever reads on ``facecolor``.
        linestyle: Outline style, e.g. ``"--"`` for a dashed frame around a group.
        hatch: Hatch pattern of the fill, e.g. ``"//"``.
        zorder: Drawing order of the box; its text draws above.
        **text_kw: Forwarded to :meth:`~matplotlib.axes.Axes.text`, e.g. ``fontsize`` or
            ``fontweight``.

    Returns:
        The box patch.
    """
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={rounding}" if rounding > 0 else "square,pad=0",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        hatch=hatch,
        zorder=zorder,
        gid="box",
    )
    ax.add_patch(box)
    if text:
        if textcolor is None:
            textcolor = mpl.rcParams["text.color"] if facecolor == "none" else _ink(facecolor)
        ax.text(x, y, text, ha="center", va="center", color=textcolor, zorder=zorder + 2, **text_kw)
    return box


def diagram_arrow(
    ax: Axes,
    tail: tuple[float, float],
    head: tuple[float, float],
    *,
    width: float = 1.0,
    color: Any = None,
    head_width: float | None = None,
    head_length: float | None = None,
    style: Literal["filled", "line"] = "filled",
    via: Sequence[tuple[float, float]] = (),
    linestyle: Any = "-",
    label: str = "",
    label_offset: tuple[float, float] = (0.0, 0.0),
    zorder: float = 3.0,
    label_kw: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> FancyArrowPatch:
    """Draw an arrow from ``tail`` to ``head``, its shaft ``width`` points wide.

    Widths are physical, so arrows keep their shape on axes of any aspect. A ``"filled"``
    arrow is one solid shape, as a flux whose width carries a value; a ``"line"`` arrow is a
    stroked line with a filled head, the connector of a flowchart. With ``via`` the arrow
    runs through those points with right-angled or any other corners, and only its last
    segment carries the head. Either way the tip ends exactly on ``head``.

    Args:
        ax: Axes to draw on.
        tail: Start, in data coordinates.
        head: Tip, in data coordinates.
        width: Shaft width (pt); the line width of a ``"line"`` arrow.
        color: Fill colour; the text colour by default.
        head_width: Head width (pt); by default ``max(3, 2.4 * width)`` for a filled arrow
            and ``max(2.4, 3 * width)`` for a line arrow.
        head_length: Head length (pt); by default ``max(2.6, 1.5 * width)`` for a filled
            arrow and ``max(2.4, 3 * width)`` for a line arrow.
        style: ``"filled"`` or ``"line"``.
        via: Points the arrow passes through between ``tail`` and ``head``, in data
            coordinates.
        linestyle: Line style of a ``"line"`` arrow's shaft, e.g. ``"--"``.
        label: Text at the midpoint of the arrow's last segment.
        label_offset: ``(dx, dy)`` of the label from the midpoint (pt).
        zorder: Drawing order; the label draws above.
        label_kw: Forwarded to :meth:`~matplotlib.axes.Axes.annotate` for the label,
            e.g. ``ha``, ``fontsize`` or ``color``.
        **kwargs: Forwarded to :class:`~matplotlib.patches.FancyArrowPatch`, e.g.
            ``connectionstyle="arc3,rad=0.2"`` for a curved arrow, ``gid``, or
            ``joinstyle="round"`` and ``capstyle="round"`` for a softer line arrow
            (mitred and butt by default).

    Returns:
        The patch of the arrow's last segment, which carries the head.

    Raises:
        ValueError: If ``style`` is neither ``"filled"`` nor ``"line"``.
    """
    if style not in ("filled", "line"):
        raise ValueError(f"style must be 'filled' or 'line', got {style!r}")
    color = mpl.rcParams["text.color"] if color is None else color
    if via:
        corners = [tail, *via]
        xs, ys = zip(*corners, strict=True)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            zorder=zorder,
        )
        tail = corners[-1]
    if style == "filled":
        hw = max(3.0, 2.4 * width) if head_width is None else head_width
        hl = max(2.6, 1.5 * width) if head_length is None else head_length
        arrowstyle = ArrowStyle.Simple(head_length=hl, head_width=hw, tail_width=width)
        line_kw: dict[str, Any] = {"linewidth": 0.0}
    else:
        hl = max(2.4, 3.0 * width) if head_length is None else head_length
        hw = hl if head_width is None else head_width
        # Matplotlib's "-|>" head width is measured from the shaft to one side.
        arrowstyle = ArrowStyle("-|>", head_length=hl, head_width=hw / 2)
        line_kw = {
            "linewidth": width,
            "linestyle": linestyle,
            "joinstyle": "miter",
            "capstyle": "butt",
        }
    patch = FancyArrowPatch(
        tail,
        head,
        arrowstyle=arrowstyle,
        mutation_scale=1.0,
        shrinkA=0.0,
        shrinkB=0.0,
        color=color,
        zorder=zorder,
        **{**line_kw, **kwargs},
    )
    ax.add_patch(patch)
    if label:
        mid = ((tail[0] + head[0]) / 2, (tail[1] + head[1]) / 2)
        kw = {"ha": "center", "va": "center", "zorder": zorder + 1, **(label_kw or {})}
        ax.annotate(label, mid, xytext=label_offset, textcoords="offset points", **kw)
    return patch


def flux_diagram(
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    vertical: ArrayLike | None = None,
    horizontal: ArrayLike | None = None,
    lateral: ArrayLike | None = None,
    ground: ArrayLike | None = None,
    parts: Mapping[str, ArrayLike] | None = None,
    notes: ArrayLike | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ground_label: str = "Surface",
    col_title: str = "",
    cmap: str | mcolors.Colormap = "BrBG_r",
    norm: mcolors.Normalize | None = None,
    colors: Mapping[str, str] | None = None,
    part_colors: Mapping[str, str] | None = None,
    reference: Mapping[str, float] | None = None,
    max_width: float = 4.5,
    box_size: tuple[float, float] = (0.48, 0.44),
    fmt: Callable[[float], str] = _fmt,
    width: float | str = "double",
) -> tuple[Axes, FluxDiagram]:
    """Draw a grid of boxes coloured by ``values``, with an arrow for every flux between them.

    Row ``k`` of every array is a layer, the first at the bottom, and column ``j`` sits at
    ``x = j + 0.5``, row ``k`` at ``y = k`` in data coordinates, so callers can annotate any
    box. Each box is filled from ``cmap`` by its value, which it prints in bold. Every flux is
    an arrow pointing the way the flux goes and labelled with its value; its width is
    ∝ √|value|, the ``reference`` value of its kind drawn ``max_width`` points wide, so widths
    are physical and independent of the axes' aspect. Non-finite fluxes are not drawn.

    Args:
        values: Box values, shape ``(n_rows, n_cols)``, e.g. each box's storage change.
        ax: Axes to draw on; a new figure is created if ``None``. The axes are turned off.
        vertical: Flux from row ``k`` up into row ``k + 1``, shape ``(n_rows - 1, n_cols)``;
            positive points up.
        horizontal: Flux from column ``j`` into column ``j + 1``, shape
            ``(n_rows, n_cols - 1)``; positive points right.
        lateral: Net inflow through a box's sides, shape ``(n_rows, n_cols)``, e.g. a
            convergence. Drawn on both sides of the box, pointing in when positive and out
            when negative, and printed inside the box under its value. Excludes
            ``horizontal``, which draws in the same gaps.
        ground: Flux from the ground up into row 0, shape ``(n_cols,)``; positive points up.
            Draws a hatched ground line under every column.
        parts: ``{key: values}`` splitting ``vertical``, each of its shape; printed left of
            every vertical arrow as ``"key value"``, one line each.
        notes: Text per box, shape ``(n_rows, n_cols)``, printed small under its lower left
            corner; empty strings print nothing.
        row_labels: One label per row, bottom first, left of the boxes.
        col_labels: One label per column, under the ground (or the bottom boxes).
        ground_label: Label of the ground line, left of it.
        col_title: Label of the column labels, left of them.
        cmap: Box colormap; a diverging one whose ends are not the arrow hues.
        norm: Box norm; by default centred on 0 and symmetric about the largest ``|values|``.
            Pass one to share colours between diagrams.
        colors: Colour per kind of flux, overriding :data:`FLUX_COLORS`.
        part_colors: Colour per key of ``parts``; ``Dark2`` in order by default.
        reference: Value drawn ``max_width`` wide, per kind of flux; by default the largest
            ``|value|`` of that kind in this call. Pass it to share widths between diagrams.
        max_width: Shaft width (pt) of the reference value.
        box_size: Box ``(width, height)`` in data units; one column is one unit wide and one
            row one unit tall.
        fmt: Formats every printed number; three significant digits by default.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.

    Returns:
        ``(ax, diagram)``; ``diagram`` holds the colour mappable, the boxes and the arrows.

    Raises:
        ValueError: An array has the wrong shape, or both ``lateral`` and ``horizontal``
            are given.
    """
    th = get_theme()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must have shape (n_rows, n_cols), got {values.shape}")
    n_rows, n_cols = values.shape
    vertical = _as_grid(vertical, (n_rows - 1, n_cols), "vertical")
    horizontal = _as_grid(horizontal, (n_rows, n_cols - 1), "horizontal")
    lateral = _as_grid(lateral, (n_rows, n_cols), "lateral")
    ground = _as_grid(ground, (n_cols,), "ground")
    if lateral is not None and horizontal is not None:
        raise ValueError("lateral and horizontal both draw between the columns; give one")
    parts = {k: _as_grid(v, (n_rows - 1, n_cols), f"parts[{k!r}]") for k, v in (parts or {}).items()}
    if parts and vertical is None:
        raise ValueError("parts split the vertical flux; give vertical too")
    note_text = None if notes is None else np.asarray(notes, dtype=object)
    if note_text is not None and note_text.shape != (n_rows, n_cols):
        raise ValueError(f"notes must have shape {(n_rows, n_cols)}, got {note_text.shape}")
    for name, labels, n in (("row_labels", row_labels, n_rows), ("col_labels", col_labels, n_cols)):
        if labels is not None and len(labels) != n:
            raise ValueError(f"{name} must have {n} entries, got {len(labels)}")

    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=min(1.2, 0.62 * (n_rows + 0.9) / n_cols))

    kind_color = {**FLUX_COLORS, **(colors or {})}
    dark2 = mpl.colormaps["Dark2"].colors
    pc = {k: mcolors.to_hex(dark2[i % len(dark2)]) for i, k in enumerate(parts)}
    pc.update(part_colors or {})
    cm = resolve_cmap(cmap)
    if norm is None:
        finite = values[np.isfinite(values)]
        limit = float(np.abs(finite).max()) if finite.size else 1.0
        norm = diverging_norm(-(limit or 1.0), limit or 1.0)
    refs = {
        "vertical": _reference(vertical, None),
        "horizontal": _reference(horizontal, None),
        "lateral": _reference(lateral, None),
        "ground": _reference(ground, None),
    }
    refs.update(reference or {})
    diagram = FluxDiagram(
        ax=ax,
        mappable=ScalarMappable(norm=norm, cmap=cm),
        boxes=np.empty((n_rows, n_cols), dtype=object),
        colors=kind_color,
        part_colors=pc,
    )
    bw, bh = box_size
    text = mpl.rcParams["text.color"]
    fs_value, fs_flux = th.fs_body, th.fs_annot
    fs_small = round(0.78 * th.fs_annot, 2)

    def arrow(kind: str, index: tuple[int, ...], value: float, a: tuple, b: tuple) -> float:
        """Draw ``value`` from ``a`` to ``b`` if positive, from ``b`` to ``a`` if not."""
        tail, head = (a, b) if value >= 0 else (b, a)
        ref = refs[kind]
        frac = np.sqrt(min(1.0, abs(value) / ref)) if ref > 0 else 0.0
        w = _MIN_WIDTH + (max_width - _MIN_WIDTH) * frac
        head_width = max(3.0, 2.4 * w)
        patch = diagram_arrow(
            ax, tail, head, width=w, head_width=head_width, color=kind_color[kind], gid=kind
        )
        diagram.arrows.append(FluxArrow(kind, index, float(value), tail, head, w, patch))
        return head_width

    def beside(x: float, y: float, label: str, clearance: float, **kw: Any) -> None:
        """``label`` centred on ``y``, ``clearance`` points right of ``x`` (left if negative)."""
        ax.annotate(
            label,
            (x, y),
            xytext=(clearance, 0.0),
            textcoords="offset points",
            ha="left" if clearance > 0 else "right",
            va="center",
            zorder=4,
            **kw,
        )

    def flux_label(x: float, y: float, value: float, kind: str, **kw: Any) -> None:
        ax.text(x, y, fmt(value), fontsize=fs_flux, fontweight="bold", color=kind_color[kind], zorder=4, **kw)

    for k in range(n_rows):
        for j in range(n_cols):
            x, v = j + 0.5, values[k, j]
            fill = cm(norm(v)) if np.isfinite(v) else "none"
            diagram.boxes[k, j] = diagram_box(ax, (x, k), bw, bh, facecolor=fill)
            two = lateral is not None
            if np.isfinite(v):
                ax.text(
                    x,
                    k + (0.085 if two else 0.0),
                    fmt(v),
                    ha="center",
                    va="center",
                    fontsize=fs_value,
                    fontweight="bold",
                    color=_ink(fill),
                    zorder=4,
                )
            if note_text is not None and str(note_text[k, j] or ""):
                ax.text(
                    x - 0.07,
                    k - bh / 2 - 0.11,
                    str(note_text[k, j]),
                    ha="right",
                    va="center",
                    fontsize=fs_small,
                    color=text,
                    alpha=0.75,
                    zorder=4,
                )
            if lateral is None or not np.isfinite(lateral[k, j]):
                continue
            h = lateral[k, j]
            ax.text(
                x,
                k - 0.095,
                fmt(h),
                ha="center",
                va="center",
                fontsize=fs_flux,
                color=kind_color["lateral"],
                zorder=4,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
            for side in (-1.0, 1.0):
                edge = x + side * (bw / 2 + 0.015)
                arrow("lateral", (k, j), h, (edge + side * _LATERAL_SPAN, k), (edge, k))

    if vertical is not None:
        for k in range(n_rows - 1):
            for j in range(n_cols):
                value = vertical[k, j]
                if not np.isfinite(value):
                    continue
                x, y0, y1 = j + 0.5, k + bh / 2 + _GAP, k + 1 - bh / 2 - _GAP
                arrow("vertical", (k, j), value, (x, y0), (x, y1))
                clearance = diagram.arrows[-1].width / 2 + 4.0
                bold = {"fontsize": fs_flux, "fontweight": "bold", "color": kind_color["vertical"]}
                beside(x, (y0 + y1) / 2, fmt(value), clearance, **bold)
                lines = [(key, p[k, j]) for key, p in parts.items()]
                step = 0.085
                for i, (key, p) in enumerate(lines):
                    y = (y0 + y1) / 2 + ((len(lines) - 1) / 2 - i) * step
                    beside(x, y, f"{key} {fmt(p)}", -clearance, fontsize=fs_small, color=pc[key])

    if horizontal is not None:
        for k in range(n_rows):
            for j in range(n_cols - 1):
                value = horizontal[k, j]
                if not np.isfinite(value):
                    continue
                x0, x1 = j + 0.5 + bw / 2 + _GAP, j + 1.5 - bw / 2 - _GAP
                arrow("horizontal", (k, j), value, (x0, k), (x1, k))
                flux_label((x0 + x1) / 2, k + 0.05, value, "horizontal", ha="center", va="bottom")

    floor = -bh / 2
    if ground is not None:
        floor = _GROUND_Y
        for j in range(n_cols):
            ax.add_patch(
                Rectangle(
                    (j + 0.06, _GROUND_Y - 0.05),
                    0.88,
                    0.05,
                    facecolor="none",
                    edgecolor="0.6",
                    hatch="//////",
                    linewidth=0.0,
                    zorder=1,
                    gid="ground_hatch",
                )
            )
            ax.plot([j + 0.06, j + 0.94], [_GROUND_Y, _GROUND_Y], color="0.55", lw=0.5, zorder=1)
            value = ground[j]
            if not np.isfinite(value):
                continue
            x, y0, y1 = j + 0.5, _GROUND_Y + 0.03, -bh / 2 - _GAP
            head_width = arrow("ground", (j,), value, (x, y0), (x, y1))
            label = {"fontsize": fs_flux, "fontweight": "bold", "color": text, "alpha": 0.7}
            beside(x, (y0 + y1) / 2, fmt(value), head_width / 2 + 2.0, **label)
        if ground_label:
            ax.text(-0.05, _GROUND_Y + 0.02, ground_label, ha="right", va="center", fontweight="bold")

    label_y = floor - _LABEL_DROP
    for j, name in enumerate(col_labels or []):
        ax.text(j + 0.5, label_y, name, ha="center", va="center", fontweight="bold")
    if col_labels and col_title:
        ax.text(-0.05, label_y, col_title, ha="right", va="center", style="italic")
    for k, name in enumerate(row_labels or []):
        ax.text(-0.05, k, name, ha="right", va="center", fontweight="bold", linespacing=1.25)

    ax.set_xlim(0.0, n_cols)
    ax.set_ylim(label_y - (0.1 if col_labels else 0.0), n_rows - 1 + bh / 2 + 0.03)
    ax.set_axis_off()
    return ax, diagram


def flux_legend(
    diagram: FluxDiagram,
    labels: Mapping[str, str],
    *,
    parts: Mapping[str, str] | None = None,
    target: Axes | Figure | None = None,
    **kwargs: Any,
) -> Legend:
    """Add a legend explaining a :func:`flux_diagram`'s colours.

    Args:
        diagram: The diagram to explain.
        labels: ``{kind: label}`` for ``"box"`` (a swatch from the box colormap) and the
            kinds of flux (a line in its colour), listed in that order.
        parts: ``{key: label}`` for the parts of the vertical flux, each shown as its key
            in its colour.
        target: Axes or figure to put the legend on; the diagram's axes by default.
        **kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.legend`, e.g. ``loc``,
            ``ncol`` or ``bbox_to_anchor``.

    Returns:
        The legend.

    Raises:
        KeyError: A label names an unknown kind, or a part has no colour.
    """
    th = get_theme()
    lw = 2.2
    handles: list[Any] = []
    for kind in ("box", *_KINDS):
        if kind not in labels:
            continue
        if kind == "box":
            m = diagram.mappable
            fill = m.to_rgba(0.6 * m.norm.vmax)
            handles.append(
                legend_handle(labels[kind], "band", facecolor=fill, edgecolor="0.55", linewidth=0.5)
            )
        else:
            handles.append(legend_handle(labels[kind], "line", color=diagram.colors[kind], lw=lw))
    unknown = set(labels) - {"box", *_KINDS}
    if unknown:
        raise KeyError(f"unknown kinds {sorted(unknown)}; expected 'box' or one of {_KINDS}")
    for key, label in (parts or {}).items():
        handles.append(
            legend_handle(label, "marker", color=diagram.part_colors[key], marker=f"${key}$", markersize=5)
        )
    kw = {"frameon": False, "fontsize": th.fs_legend, "handlelength": 1.6, **kwargs}
    return (target or diagram.ax).legend(handles=handles, **kw)
