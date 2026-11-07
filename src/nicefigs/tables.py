"""Scorecards: model × metric tables rendered as coloured-cell figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from .palettes import cmap as resolve_cmap
from .style import get_theme
from .units import DIRECTIONS, check_direction, prettify_unit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from numpy.typing import ArrayLike

#: Cell side length (inches) used to size the figure when ``square=True``.
_SQUARE_CELL_INCH = 0.34


def skill_gap(
    values: ArrayLike,
    reference: ArrayLike,
    *,
    uncertainty: ArrayLike | None = None,
    direction: str | Sequence[str] = "higher_better",
    target: ArrayLike = 0.0,
) -> np.ndarray:
    """Signed distance of ``values`` from a reference, optionally in units of its uncertainty.

    Positive is always better, whichever way a metric is read, so one diverging
    colour scale can carry metrics in different units (see :func:`scorecard`).
    A ``"target"`` metric is scored by closeness to ``target``:
    ``|reference - target| - |values - target|``. Dividing by ``uncertainty``
    turns each gap into a signal-to-noise ratio: ``-3`` is three error bars
    worse than the reference, ``0`` indistinguishable from it.

    Args:
        values: ``(n_rows, n_cols)`` metric table; a 1-D input is one row.
        reference: Values to measure against, broadcast against ``values``.
        uncertainty: Scale the gap is divided by, in the metric's units (e.g. a
            standard error), broadcast against ``values``. Non-positive or
            non-finite entries give ``nan``. ``None`` keeps the metric's units.
        direction: ``"higher_better"``, ``"lower_better"`` or ``"target"``,
            for the whole table or one per column.
        target: Value a ``"target"`` metric is best at, broadcast against ``values``.

    Returns:
        ``(n_rows, n_cols)`` signed gaps, positive where ``values`` beat
        ``reference``; ``nan`` wherever an input is missing.

    Raises:
        ValueError: ``values`` and ``reference`` do not broadcast, a per-column
            ``direction`` does not label every column, or a direction is not in
            :data:`DIRECTIONS`.
    """
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
    for j, how in enumerate(directions):
        if how == "higher_better":
            gap[:, j] = values[:, j] - reference[:, j]
        elif how == "lower_better":
            gap[:, j] = reference[:, j] - values[:, j]
        else:  # closeness to the target, so positive is still better
            gap[:, j] = np.abs(reference[:, j] - target[:, j]) - np.abs(values[:, j] - target[:, j])

    if uncertainty is not None:
        scale = np.broadcast_to(np.asarray(uncertainty, dtype=float), values.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(np.isfinite(scale) & (scale > 0), gap / scale, np.nan)
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


def scorecard(
    values: ArrayLike,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    ax: Axes | None = None,
    direction: Mapping[str, str] | None = None,
    per_column_norm: bool = True,
    highlight_best: bool = True,
    reference: ArrayLike | None = None,
    uncertainty: ArrayLike | None = None,
    target: ArrayLike = 0.0,
    gap_limit: float = 3.0,
    cell_format: str = "{:.3g}",
    col_units: Mapping[str, str] | None = None,
    cmap: str | Colormap = "RdYlGn",
    na_color: Any = "#eeeeee",
    width: float | str = "half",
    title: str = "",
    norm: Normalize | None = None,
    square: bool = False,
    col_groups: Sequence[str] | None = None,
    group_gap: float = 0.6,
    colorbar: bool = False,
    colorbar_label: str = "",
    colorbar_extend: str | None = None,
) -> Axes:
    """Render a ``(n_rows, n_cols)`` metric table as a coloured-cell figure.

    By default each cell is coloured by its per-column goodness rank (green is
    best) and the best cell per column is outlined. ``reference`` colours cells
    by :func:`skill_gap` instead, and ``norm`` by their raw value.

    Args:
        values: Table of shape ``(len(row_labels), len(col_labels))``; ``nan``
            cells are drawn blank in ``na_color``.
        row_labels: Row (model) names.
        col_labels: Column (metric) names.
        ax: Axes to draw on; a new figure is created if ``None``.
        direction: ``{col_label: direction}`` with values from
            :data:`~nicefigs.units.DIRECTIONS`; unlisted columns are
            ``"higher_better"``. ``"target"`` ranks by closeness to ``target``.
        per_column_norm: Rank each column on its own min→max (else one global
            range). Ignored for colour when ``norm`` or ``reference`` is given.
        highlight_best: Outline the best cell in each column by the goodness rank.
        reference: Values to measure against, broadcast against ``values``.
            Cell colour becomes the signed :func:`skill_gap` on one diverging
            scale; cell text stays the raw value.
        uncertainty: With ``reference``, the scale each gap is divided by.
        target: The value a ``"target"`` metric is best at, broadcast against
            ``values``.
        gap_limit: Gap at which the ``reference`` colour scale saturates.
        cell_format: Format string for the cell text.
        col_units: ``{col_label: unit}`` shown below the column name.
        cmap: Colormap name or object.
        na_color: Colour of missing cells.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        title: Left-aligned title.
        norm: Normalisation mapping raw ``values`` straight to ``cmap`` (e.g. from
            :func:`nicefigs.scorecard_diverging_cmap`).
        square: Size the figure from the cell count and force square cells;
            only sizes the figure when ``ax`` is ``None``.
        col_groups: One group name per column; consecutive equal names form a
            group with a gap before it and a bold title above it.
        group_gap: Gap before each group, in cell widths.
        colorbar: Add a horizontal colorbar below the table.
        colorbar_label: Colorbar label.
        colorbar_extend: ``"neither"``, ``"min"``, ``"max"`` or ``"both"``;
            inferred from ``cmap``'s under/over colours if ``None``.

    Returns:
        The axes.
    """
    th = get_theme()
    values = np.asarray(values, dtype=float)
    nr, nc = values.shape
    directions = [check_direction((direction or {}).get(c, "higher_better")) for c in col_labels]
    targets = np.broadcast_to(np.asarray(target, dtype=float), values.shape)
    cm = resolve_cmap(cmap)

    lefts = np.zeros(nc)
    cur = 0.0
    for j in range(nc):
        if col_groups is not None and j > 0 and col_groups[j] != col_groups[j - 1]:
            cur += group_gap
        lefts[j] = cur
        cur += 1.0
    total_w = cur

    if ax is None:
        from .figure import subplots

        if square:
            fig_w = max(total_w * _SQUARE_CELL_INCH, 1.0)
            fig_h = max(nr * _SQUARE_CELL_INCH, 1.0)
            _, ax = subplots(width=fig_w, height=fig_h)
        else:
            _, ax = subplots(width=width, aspect=max(0.3, 0.12 * nr + 0.12))
    if square:
        ax.set_aspect("equal")

    # Goodness in [0, 1] (1 = best); drives `highlight_best` whatever colours the cells.
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

    gap = None
    if reference is not None:
        gap = skill_gap(
            values,
            reference,
            uncertainty=uncertainty,
            direction=directions,
            target=target,
        )
        norm = (
            norm if norm is not None else mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-gap_limit, vmax=gap_limit)
        )

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, nr)
    ax.invert_yaxis()

    for i in range(nr):
        for j in range(nc):
            v = values[i, j]
            if not np.isfinite(v):
                face = na_color
            elif gap is not None:
                face = na_color if not np.isfinite(gap[i, j]) else cm(norm(gap[i, j]))
            elif norm is not None:
                face = cm(norm(v))
            else:
                face = cm(float(np.clip(good[i, j], 0, 1)))
            ax.add_patch(
                mpatches.Rectangle((lefts[j], i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.0)
            )
            if np.isfinite(v):
                r, g, b = face[:3] if isinstance(face, tuple) else (0.5, 0.5, 0.5)
                txtcol = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#222222"
                ax.text(
                    lefts[j] + 0.5,
                    i + 0.5,
                    cell_format.format(v),
                    ha="center",
                    va="center",
                    fontsize=th.fs_annot,
                    color=txtcol,
                )

    if highlight_best:
        for j in range(nc):
            col_good = good[:, j].copy()
            col_good[~np.isfinite(values[:, j])] = -np.inf
            if np.isfinite(col_good).any():
                bi = int(np.argmax(col_good))
                ax.add_patch(
                    mpatches.Rectangle(
                        (lefts[j], bi), 1, 1, fill=False, edgecolor=th.spine_color, linewidth=1.6, zorder=5
                    )
                )

    ax.set_xticks(lefts + 0.5)
    heads = []
    for c in col_labels:
        u = (col_units or {}).get(c)
        heads.append(f"{c}\n[{prettify_unit(u)}]" if u else c)
    ax.set_xticklabels(heads, fontsize=th.fs_tick)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=th.fs_tick)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if col_groups is not None:
        import matplotlib.transforms as mtransforms

        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        j = 0
        while j < nc:
            k = j
            while k + 1 < nc and col_groups[k + 1] == col_groups[j]:
                k += 1
            cx = 0.5 * (lefts[j] + lefts[k] + 1)
            ax.text(
                cx,
                1.22,
                col_groups[j],
                transform=blend,
                ha="center",
                va="bottom",
                fontsize=th.fs_tick,
                fontweight="bold",
            )
            j = k + 1

    if title:
        ax.set_title(title, fontsize=th.fs_title, pad=28 if col_groups is not None else 14, loc="left")

    if colorbar:
        from matplotlib.cm import ScalarMappable

        from .colorbars import colorbar as _draw_colorbar

        cb_norm = norm if norm is not None else mcolors.Normalize(vmin=0.0, vmax=1.0)
        extend = colorbar_extend if colorbar_extend is not None else _cmap_extend(cm)
        sm = ScalarMappable(norm=cb_norm, cmap=cm)
        sm.set_array([])
        _draw_colorbar(sm, ax, location="bottom", extend=extend, label=colorbar_label)

    return ax
