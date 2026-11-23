"""Scorecards: model × metric tables rendered as coloured-cell figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from ._scorecard import (
    FS_CBAR,
    FS_CBAR_TICK,
    FS_GROUP,
    FS_HEAD,
    FS_ROW,
    FS_UNIT,
    FS_VALUE,
    NA_TEXT,
    draw_colorbar,
    draw_header,
    fit_headings,
    fit_labels,
    group_blocks,
    header_heights,
    text_width_pt,
)
from .figure import figure_size
from .palettes import cmap as resolve_cmap
from .palettes import scorecard_skill_cmap as resolve_skill_cmap
from .panels import outline_boxes
from .style import get_theme
from .units import DIRECTIONS, check_direction, prettify_unit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from numpy.typing import ArrayLike
#: Cell side (inches) of a scorecard figure created without ``width`` or ``cell_in``.
_DEFAULT_CELL_IN = 0.42


def skill_gap(
    values: ArrayLike,
    reference: ArrayLike,
    *,
    uncertainty: ArrayLike | None = None,
    direction: str | Sequence[str] = "higher_better",
    target: ArrayLike = 0.0,
    mode: str = "gap",
    floor: float = 0.0,
) -> np.ndarray:
    """Signed distance of ``values`` from a reference: raw, in units of its uncertainty, or in percent.

    Positive is always better, whichever way a metric is read, so one diverging
    colour scale can carry metrics in different units (see :func:`scorecard`).
    A ``"target"`` metric is scored by closeness to ``target``:
    ``|reference - target| - |values - target|``. Dividing by ``uncertainty``
    turns each gap into a signal-to-noise ratio: ``-3`` is three error bars
    worse than the reference, ``0`` indistinguishable from it. With ``mode="percent"``
    the gap is ``100 * gap / max(|reference|, floor)`` (``|reference - target|`` for a
    ``"target"`` metric): the percent a value is better than its reference.

    Args:
        values: ``(n_rows, n_cols)`` metric table; a 1-D input is one row.
        reference: Values to measure against, broadcast against ``values``.
        uncertainty: Scale the gap is divided by, in the metric's units (e.g. a
            standard error), broadcast against ``values``. Non-positive or
            non-finite entries give ``nan``. ``None`` keeps the metric's units.
        direction: ``"higher_better"``, ``"lower_better"`` or ``"target"``,
            for the whole table or one per column.
        target: Value a ``"target"`` metric is best at, broadcast against ``values``.
        mode: ``"gap"`` for the raw (or standardised) gap, ``"percent"`` for the gap relative
            to the reference.
        floor: With ``mode="percent"``, smallest denominator, in the metric's units, so a
            reference near zero does not blow the percent up. Zero denominators give ``nan``.

    Returns:
        ``(n_rows, n_cols)`` signed gaps, positive where ``values`` beat
        ``reference``; ``nan`` wherever an input is missing.

    Raises:
        ValueError: ``values`` and ``reference`` do not broadcast, a per-column
            ``direction`` does not label every column, a direction is not in
            :data:`DIRECTIONS`, ``mode`` is unknown, or ``mode="percent"`` is combined
            with ``uncertainty``.
    """
    if mode not in ("gap", "percent"):
        raise ValueError(f"unknown mode {mode!r}; known: ['gap', 'percent']")
    if mode == "percent" and uncertainty is not None:
        raise ValueError("mode='percent' cannot be combined with an uncertainty")
    values = np.atleast_2d(np.asarray(values, dtype=float))
    try:
        reference = np.broadcast_to(np.asarray(reference, dtype=float), values.shape)
        target = np.broadcast_to(np.asarray(target, dtype=float), values.shape)
    except ValueError:
        raise ValueError(f"values {values.shape} and reference must broadcast to the same shape") from None

    directions = [direction] * values.shape[1] if isinstance(direction, str) else list(direction)
    if len(directions) != values.shape[1]:
        raise ValueError(f"give one direction per column: {values.shape[1]} columns, {len(directions)} given")
    unknown = sorted({d for d in directions if d not in DIRECTIONS})
    if unknown:
        raise ValueError(f"unknown direction(s) {unknown}; known: {list(DIRECTIONS)}")

    gap = np.full(values.shape, np.nan)
    scale = np.abs(reference).copy()
    for j, how in enumerate(directions):
        if how == "higher_better":
            gap[:, j] = values[:, j] - reference[:, j]
        elif how == "lower_better":
            gap[:, j] = reference[:, j] - values[:, j]
        else:  # closeness to the target, so positive is still better
            gap[:, j] = np.abs(reference[:, j] - target[:, j]) - np.abs(values[:, j] - target[:, j])
            scale[:, j] = np.abs(reference[:, j] - target[:, j])

    if mode == "percent":
        denom = np.maximum(scale, floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, 100.0 * gap / denom, np.nan)
    if uncertainty is not None:
        unc = np.broadcast_to(np.asarray(uncertainty, dtype=float), values.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(np.isfinite(unc) & (unc > 0), gap / unc, np.nan)
    return gap


def _cmap_extend(cm: Colormap) -> str:
    """Infer a matplotlib ``extend`` mode from a colormap's under/over colours."""
    has_under = getattr(cm, "_rgba_under", None) is not None
    has_over = getattr(cm, "_rgba_over", None) is not None
    if has_under and has_over:
        return "both"
    if has_under:
        return "min"
    if has_over:
        return "max"
    return "neither"


def _group_offsets(groups: Sequence[str] | None, n: int, gap: float) -> tuple[np.ndarray, float]:
    """Start of each of ``n`` unit cells along an axis, with ``gap`` before each new group."""
    starts = np.zeros(n)
    cur = 0.0
    for k in range(n):
        if groups is not None and k > 0 and groups[k] != groups[k - 1]:
            cur += gap
        starts[k] = cur
        cur += 1.0
    return starts, cur


def _as_mask(mask: ArrayLike | None, shape: tuple[int, int], name: str) -> np.ndarray:
    """Boolean ``shape`` mask from an optional array-like."""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def _is_missing(entry: Any) -> bool:
    """Whether a printed entry is absent (a ``nan`` or ``None``)."""
    if entry is None:
        return True
    if isinstance(entry, str):
        return False
    return not np.isfinite(float(entry))


def scorecard(
    values: ArrayLike,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    ax: Axes | None = None,
    direction: Mapping[str, str] | None = None,
    per_column_norm: bool = True,
    highlight_best: bool = False,
    reference: ArrayLike | None = None,
    uncertainty: ArrayLike | None = None,
    target: ArrayLike = 0.0,
    gap_limit: float = 3.0,
    floor: float = 0.0,
    color_values: ArrayLike | None = None,
    text: ArrayLike | None = None,
    reference_row: str | int | None = None,
    reference_cells: ArrayLike | None = None,
    unresolved: ArrayLike | None = None,
    baseline_suffix: str = " (baseline)",
    cell_format: str = "{:.3g}",
    na_text: str = "n/a",
    col_units: Mapping[str, str] | None = None,
    cmap: str | Colormap | None = None,
    na_color: Any = "#e8e8e8",
    width: float | str | None = None,
    cell_in: float | None = None,
    font_scale: float = 1.0,
    title: str = "",
    norm: Normalize | None = None,
    square: bool = False,
    col_groups: Sequence[str] | None = None,
    group_gap: float = 0.35,
    group_rule: bool = True,
    row_groups: Sequence[str] | None = None,
    row_group_gap: float = 0.35,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_extend: str | None = None,
) -> Axes:
    """Render a ``(n_rows, n_cols)`` metric table as a coloured-cell scorecard.

    Cell text is the raw value; cell colour is the *percent better than a baseline* on the
    fixed 11-band scale of :func:`~nicefigs.scorecard_skill_cmap` (blue better, red worse). The
    baseline is given as a row (``reference_row``), as one outlined cell per column
    (``reference_cells``), as an array (``reference``) or as ready colour values
    (``color_values``). Missing values print ``na_text``; ``unresolved`` cells print their
    value in grey. Without any baseline, cells are coloured by per-column goodness rank on a
    continuous version of the same palette. Columns and rows can be grouped with a gap;
    column groups get a title and a rule.

    A ``cmap`` given by name resolves through :func:`~nicefigs.cmap`, where ``"RdBu"`` is
    ``RdBu_r`` (red high), so with ``reference`` the better cells come out red; pass
    ``matplotlib.colormaps["RdBu"]`` for blue better.

    Args:
        values: Raw table of shape ``(len(row_labels), len(col_labels))``; printed unless
            ``text`` is given. ``nan`` prints ``na_text`` on a grey cell.
        row_labels: Row (model) names.
        col_labels: Column (metric) names; wrapped at spaces to the cell width, or at your
            own ``"\\n"``. Headings shrink from 7 pt towards the unit size until neighbours stand
            2 pt apart.
        ax: Axes to draw on; a new figure sized from the cells is created if ``None``.
        direction: ``{col_label: direction}`` with values from
            :data:`~nicefigs.units.DIRECTIONS`; unlisted columns are ``"higher_better"``.
        per_column_norm: Rank each column on its own min to max (else one global range);
            only for the rank colouring.
        highlight_best: Outline the best cell of each column by goodness rank.
        reference: Values to measure against, broadcast against ``values``. Colour becomes
            :func:`skill_gap` in percent, or in units of ``uncertainty`` if that is given.
        uncertainty: With ``reference``, the scale each gap is divided by; the colour scale
            is then ``(-gap_limit, gap_limit)`` on the same palette.
        target: Value a ``"target"`` metric is best at, broadcast against ``values``.
        gap_limit: Standardised gap at which the colour scale saturates.
        floor: Smallest percent denominator, see :func:`skill_gap`.
        color_values: ``(n_rows, n_cols)`` numbers that colour the cells, normally percent
            better than the baseline; ``nan`` gives a grey cell. Overrides ``reference``.
        text: ``(n_rows, n_cols)`` printed entries (strings or numbers) instead of ``values``.
        reference_row: Row label or index of the baseline row. It is drawn grey with its
            raw values inside one outline, gets ``baseline_suffix`` on its label, and
            supplies ``reference`` when none is given. Without ``row_groups`` it forms its
            own row group.
        reference_cells: Boolean mask of per-column reference cells, each outlined and drawn
            grey; with one cell per column it also supplies ``reference``.
        unresolved: Boolean mask of cells too close to the baseline to resolve; drawn grey with
            the value in grey.
        baseline_suffix: Appended to the label of ``reference_row``.
        cell_format: Format string for numeric cell text.
        na_text: Text of a missing cell.
        col_units: ``{col_label: unit}`` shown as a second heading line in grey.
        cmap: Colormap name or object; default the scale of :func:`~nicefigs.scorecard_skill_cmap`.
        na_color: Colour of baseline, missing and unresolved cells.
        width: Figure width (preset name or inches) when ``ax`` is ``None``; the cell size is
            solved from it. Default: size from ``cell_in``.
        cell_in: Cell side in inches when ``ax`` is ``None`` and ``width`` is not given;
            default 0.42.
        font_scale: Factor on every font size of the card (cell text 7.5 pt, row labels 8 pt,
            headings 7 pt, units 6 pt, group titles 9.5 pt, colourbar label 8 pt).
        title: Left-aligned title above the header.
        norm: Normalisation of the colour numbers. Given alone it maps ``values`` straight to
            ``cmap``.
        square: Force square cells on a given ``ax``; a new figure has square cells anyway.
        col_groups: One group name per column; consecutive equal names form a group with a
            gap before it and a title with a rule above it.
        group_gap: Gap before each column group, in cell widths.
        group_rule: Draw the rule under column group titles.
        row_groups: One group name per row; consecutive equal names form a group with a gap
            before it.
        row_group_gap: Gap before each row group, in cell heights.
        colorbar: Add a horizontal colourbar below the grid, about half its width, with ticks at
            the band edges and pointed ends.
        colorbar_label: Colourbar label; default "% better than the baseline" (or "... than the
            outlined reference of the column" with ``reference_cells`` alone).
        colorbar_extend: ``"neither"``, ``"min"``, ``"max"`` or ``"both"``; inferred from
            ``cmap``'s under/over colours if ``None``.

    Returns:
        The axes; one unit is one cell, y grows downwards, and cells are the filled patches.

    Raises:
        ValueError: A table or mask has the wrong shape, ``reference_row`` is not a row, the
            reference cells do not give one per column, or both ``width`` and ``cell_in`` are
            given.
    """
    th = get_theme()
    values = np.asarray(values, dtype=float)
    nr, nc = values.shape
    if width is not None and cell_in is not None:
        raise ValueError("give either width or cell_in, not both")
    directions = [check_direction((direction or {}).get(c, "higher_better")) for c in col_labels]
    targets = np.broadcast_to(np.asarray(target, dtype=float), values.shape)
    labels = [str(r) for r in row_labels]

    ridx = None
    if reference_row is not None:
        ridx = labels.index(str(reference_row)) if isinstance(reference_row, str) else int(reference_row)
        if not 0 <= ridx < nr:
            raise ValueError(f"reference_row {reference_row!r} is not one of the {nr} rows")
        labels[ridx] += baseline_suffix
    ref_cells = _as_mask(reference_cells, values.shape, "reference_cells")
    is_ref = ref_cells.copy()
    if ridx is not None:
        is_ref[ridx, :] = True
    unres = _as_mask(unresolved, values.shape, "unresolved")

    # Where the colour numbers come from, and what they mean.
    kind = "rank"
    cvals = None
    if color_values is not None:
        cvals = np.asarray(color_values, dtype=float)
        if cvals.shape != values.shape:
            raise ValueError(f"color_values must have shape {values.shape}, got {cvals.shape}")
        kind = "percent"
    else:
        if reference is None and ridx is not None:
            reference = np.broadcast_to(values[ridx], values.shape)
        elif reference is None and ref_cells.any():
            if not (ref_cells.sum(axis=0) == 1).all():
                raise ValueError("reference_cells must mark exactly one cell per column")
            reference = np.broadcast_to(np.where(ref_cells, values, 0.0).sum(axis=0), values.shape)
        if reference is not None:
            cvals = skill_gap(
                values,
                reference,
                uncertainty=uncertainty,
                direction=directions,
                target=target,
                mode="gap" if uncertainty is not None else "percent",
                floor=floor,
            )
            kind = "standardised" if uncertainty is not None else "percent"
        elif norm is not None:
            cvals, kind = values, "raw"

    skill_cm, skill_norm = resolve_skill_cmap()
    if cmap is not None:
        cm = resolve_cmap(cmap)
    elif kind == "rank":
        cm = mcolors.LinearSegmentedColormap.from_list("scorecard_skill_continuous", skill_cm.colors)
    else:
        cm = skill_cm
    if norm is None and cvals is not None:
        if kind == "standardised":
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-gap_limit, vmax=gap_limit)
        elif cmap is None:
            norm = skill_norm
        else:
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-50.0, vmax=50.0)

    # Goodness in [0, 1] (1 = best); colours the rank mode and drives `highlight_best`.
    good = np.full_like(values, 0.5)
    if per_column_norm:
        for j in range(nc):
            col = values[:, j]
            dirn = directions[j]
            metric = np.abs(col - targets[:, j]) if dirn == "target" else col
            finite = metric[np.isfinite(metric)]
            if finite.size == 0:
                continue
            lo, hi = finite.min(), finite.max()
            g = np.zeros_like(metric) if hi <= lo else (metric - lo) / (hi - lo)
            if dirn in ("lower_better", "target"):
                g = 1 - g
            good[:, j] = g
    else:
        finite = values[np.isfinite(values)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
        good = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)

    printed = None
    if text is not None:
        printed = np.empty(values.shape, dtype=object)
        printed[...] = np.asarray(text, dtype=object).reshape(values.shape)

    lefts, total_w = _group_offsets(col_groups, nc, group_gap)
    row_keys = row_groups
    if row_keys is None and ridx is not None:
        row_keys = ["ref" if i == ridx else "rest" for i in range(nr)]
    tops, total_h = _group_offsets(row_keys, nr, row_group_gap)

    s = font_scale
    heads_raw = [str(c) for c in col_labels]
    units_raw = [prettify_unit((col_units or {}).get(c)) for c in col_labels]
    blocks = group_blocks(col_groups, lefts) if col_groups is not None else []

    def _fit_header(cell_pt: float):
        """Headings, units and group titles wrapped and sized to clear their neighbours, and their heights."""
        fh, heads = fit_headings(heads_raw, lefts, cell_pt, FS_HEAD * s, FS_UNIT * s)
        fu, units = fit_labels(units_raw, [(x, x + 1.0) for x in lefts], cell_pt, FS_UNIT * s, 4.5 * s)
        fg, titles = fit_labels(
            [b[0] for b in blocks], [b[1:] for b in blocks], cell_pt, FS_GROUP * s, FS_UNIT * s
        )
        hh = header_heights(
            heads,
            max((u.count("\n") + 1 for u in units if u), default=0),
            col_groups is not None,
            s,
            fh,
            unit_size=fu,
            group_lines=max((t.count("\n") + 1 for t in titles), default=1),
            group_size=fg,
        )
        return fh, heads, fu, units, fg, titles, hh

    barlabel = colorbar_label
    if barlabel is None:
        barlabel = (
            "% better than the outlined reference of the column"
            if ridx is None and ref_cells.any()
            else "% better than the baseline"
            if kind == "percent"
            else ""
        )

    if ax is None:
        from .figure import figure

        row_w_pt = max((text_width_pt(lab, FS_ROW * s) for lab in labels), default=0.0)
        left_in = (row_w_pt + 6.0 * s) / 72.0 + 0.06
        right_in = 0.12
        cell_for_wrap = cell_in if cell_in is not None else _DEFAULT_CELL_IN
        cbar_w = 0.0
        if colorbar:
            cbar_w = 2.7

        def _plan(cell: float):
            header = _fit_header(cell * 72.0)
            hh = header[-1]
            top_in = (hh["top"] + 4.0 * s) / 72.0 + (th.fs_title * 1.6 / 72.0 if title else 0.0)
            grid_w, grid_h = total_w * cell, total_h * cell
            extra = max(0.0, 0.5 * (cbar_w - grid_w)) + 0.06
            l_in, r_in = max(left_in, extra), max(right_in, extra)
            bottom_in = (0.22 + 0.11 + (FS_CBAR_TICK + FS_CBAR) * s * 1.4 / 72.0 + 0.1) if colorbar else 0.08
            return header, grid_w, grid_h, l_in, r_in, top_in, bottom_in

        if width is not None:
            w_in = figure_size(width)[0]
            cell_for_wrap = _DEFAULT_CELL_IN
            for _ in range(3):  # wrapping depends on the cell, the margins on the wrapping
                _, _, _, l_in, r_in, _, _ = _plan(cell_for_wrap)
                cell_for_wrap = max((w_in - l_in - r_in) / total_w, 0.05)
        header, grid_w, grid_h, l_in, r_in, top_in, bottom_in = _plan(cell_for_wrap)
        fig_w, fig_h = l_in + grid_w + r_in, top_in + grid_h + bottom_in
        fig = figure(width=fig_w, height=fig_h)
        ax = fig.add_axes([l_in / fig_w, bottom_in / fig_h, grid_w / fig_w, grid_h / fig_h])
        cell_pt = cell_for_wrap * 72.0
    else:
        if square:
            ax.set_aspect("equal")
        fig = ax.figure
        box = ax.get_position()
        cell_pt = max(fig.get_figwidth() * box.width * 72.0 / total_w, 1.0)
        header = _fit_header(cell_pt)
    fh, heads, fu, units, fg, titles, hh = header

    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    face_of = {}
    for i in range(nr):
        for j in range(nc):
            v = values[i, j]
            entry = printed[i, j] if printed is not None else v
            missing = _is_missing(entry)
            grey_state = missing or (unres[i, j] and not is_ref[i, j])
            if is_ref[i, j]:
                face, ink = na_color, th.spine_color
            elif grey_state or (cvals is not None and not np.isfinite(cvals[i, j])):
                face, ink = na_color, NA_TEXT
            elif cvals is not None:
                face, ink = cm(norm(cvals[i, j])), None
            else:
                face, ink = cm(float(np.clip(good[i, j], 0, 1))), None
            if ink is None:
                r, g, b = mcolors.to_rgb(face)
                ink = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else th.spine_color
            face_of[i, j] = face
            ax.add_patch(
                mpatches.Rectangle(
                    (lefts[j], tops[i]), 1, 1, facecolor=face, edgecolor="white", linewidth=1.0
                )
            )
            if missing:
                label = na_text
            elif isinstance(entry, str):
                label = entry
            else:
                label = cell_format.format(float(entry))
            ax.text(
                lefts[j] + 0.5,
                tops[i] + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=FS_VALUE * s,
                color=ink,
            )

    boxes = [
        (lefts[j], tops[i], 1.0, 1.0) for i, j in zip(*np.nonzero(ref_cells)) if ridx is None or i != ridx
    ]
    if ridx is not None:
        boxes.append((0.0, tops[ridx], total_w, 1.0))
    outline_boxes(ax, boxes, color="#222222", linewidth=2.0, zorder=6)

    if highlight_best:
        for j in range(nc):
            col_good = good[:, j].copy()
            col_good[~np.isfinite(values[:, j])] = -np.inf
            if np.isfinite(col_good).any():
                bi = int(np.argmax(col_good))
                ax.add_patch(
                    mpatches.Rectangle(
                        (lefts[j], tops[bi]),
                        1,
                        1,
                        fill=False,
                        edgecolor=th.spine_color,
                        linewidth=1.6,
                        zorder=5,
                    )
                )

    for i, lab in enumerate(labels):
        ax.annotate(
            lab,
            (0, tops[i] + 0.5),
            xytext=(-5.0 * s, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=FS_ROW * s,
            color=th.spine_color,
            annotation_clip=False,
        )

    draw_header(
        ax,
        lefts,
        heads,
        units,
        col_groups,
        hh,
        group_rule=group_rule,
        cell_pt=cell_pt,
        scale=s,
        head_size=fh,
        unit_size=fu,
        group_titles=titles,
        group_size=fg,
    )

    if title:
        ax.set_title(title, fontsize=th.fs_title, pad=hh["top"] + 4.0 * s, loc="left")

    if colorbar:
        cb_norm = norm if norm is not None else mcolors.Normalize(vmin=0.0, vmax=1.0)
        extend = colorbar_extend if colorbar_extend is not None else _cmap_extend(cm)
        grid_in = ax.get_position().width * ax.figure.get_figwidth()
        draw_colorbar(
            ax,
            cm,
            cb_norm,
            label=barlabel,
            extend=extend,
            width_in=max(0.5 * grid_in, 2.7),
            drop_in=0.22,
            thickness_in=0.11,
            scale=s,
        )

    return ax
