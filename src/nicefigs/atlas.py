"""Atlas panels: one figure that summarises a mapped quantity with its examples.

:func:`component_panel` composes a world map of a per-location quantity,
a histogram of its values, a text box with free-form notes and a label slot, and
rows of example image chips with a coarse grid of the quantity overlaid.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator

from .figure import figure_size
from .images import image, overlay_grid
from .maps import map as world_map
from .panels import clean_title
from .style import COLORS, get_theme, style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from numpy.typing import ArrayLike


def component_panel(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    *,
    chips: ArrayLike | None = None,
    overlays: ArrayLike | None = None,
    window: tuple[int, int, int] | None = None,
    hist_edges: ArrayLike | None = None,
    hist_counts: ArrayLike | None = None,
    text: Sequence[str] = (),
    label: str = "",
    title: str = "",
    map_label: str = "",
    cmap: str | Colormap = "inferno",
    overlay_cmap: str | Colormap = "inferno",
    overlay_alpha: float = 0.6,
    overlay_mask_below: float | None = None,
    extra_rows: Sequence[Mapping[str, Any]] = (),
    width: float | str = "full",
    fig: Figure | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Draw the atlas panel of one component.

    Top row: a Robinson map of ``values`` at ``lon``/``lat``, the histogram given by
    ``hist_edges``/``hist_counts``, and a text box (``text`` lines, then the label
    slot, left blank as ``label: ____`` when ``label`` is empty). Bottom row: one
    axes per chip, its RGB stretched per chip, with ``overlays[i]`` drawn over
    ``window`` by :func:`nicefigs.overlay_grid` on one shared colour scale.

    Args:
        lon: ``(n,)`` longitudes in degrees east.
        lat: ``(n,)`` latitudes in degrees north.
        values: ``(n,)`` values of the component at those locations.
        chips: ``(m, H, W, 3)`` RGB chips; ``None`` draws no bottom row.
        overlays: ``(m, h, w)`` coarse grids, one per chip.
        window: ``(row, col, size)`` of the overlaid window in chip pixels; required
            with ``overlays``.
        hist_edges: Histogram bin edges, as returned by :func:`numpy.histogram`.
        hist_counts: Histogram counts, as returned by :func:`numpy.histogram`.
        text: Lines for the text box; wrapped to fit, shrinking the font if needed.
        label: The component's short label.
        title: Panel title, above the map.
        map_label: Colorbar label of the map.
        cmap: Colormap of the map.
        overlay_cmap: Colormap of the overlays.
        overlay_alpha: Opacity of the overlays.
        overlay_mask_below: Overlay cells at or below this value are transparent.
        extra_rows: Further chip rows below the first, each a dict with ``chips``,
            optional ``overlays`` (one per chip, needs ``window``) and optional ``title``
            (shown above the row's first chip).
        width: Figure width (inches or a :data:`nicefigs.figure.WIDTHS` name).
        fig: Draw into this figure instead of a new one.

    Returns:
        ``(fig, axes)``, ``axes`` a dict with ``map``, ``hist``, ``text``, the list ``chips``
        and ``extra_rows`` (one list of axes per extra row).

    Raises:
        ValueError: If ``overlays`` come without a ``window`` or not one per chip.
    """
    n_chips = 0 if chips is None else len(chips)
    for c, o in [(chips, overlays)] + [(r["chips"], r.get("overlays")) for r in extra_rows]:
        if o is not None:
            if window is None:
                raise ValueError("overlays need the chip window they cover")
            if len(o) != (0 if c is None else len(c)):
                raise ValueError(f"{len(o)} overlays for {0 if c is None else len(c)} chips")
    th = get_theme()
    with style(th):
        if fig is None:
            aspect = (0.62 if n_chips else 0.42) + 0.2 * len(extra_rows)
            fig = plt.figure(figsize=figure_size(width, aspect=aspect), dpi=th.dpi)
        heights = [1.0] + ([0.62] if n_chips else []) + [0.62] * len(extra_rows)
        gs = fig.add_gridspec(len(heights), 12, height_ratios=heights, hspace=0.12, wspace=0.25)
        _, ax_map, _ = world_map(
            lon,
            lat,
            values,
            ax=None,
            fig=fig,
            subplot_spec=gs[0, 0:7],
            kind="scatter",
            cmap=cmap,
            inset_hist=False,
            cbar=True,
            colorbar_label=map_label,
            size=3.0,
        )
        if title:
            clean_title(ax_map, title)
        side = gs[0, 9:12].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
        ax_hist = fig.add_subplot(side[0])
        if hist_edges is not None and len(hist_edges):
            ax_hist.stairs(hist_counts, hist_edges, fill=True, color=COLORS["primary"], alpha=0.85)
        ax_hist.set_xlabel("value", fontsize=th.fs_axis_label)
        ax_hist.set_ylabel("count", fontsize=th.fs_axis_label)
        ax_hist.tick_params(labelsize=th.fs_tick)
        ax_hist.yaxis.set_major_locator(MaxNLocator(3))
        ax_hist.yaxis.set_major_formatter(
            EngFormatter(sep="")
        )  # 20k, not 20000: keeps clear of the map's colorbar
        ax_text = fig.add_subplot(side[1])
        ax_text.axis("off")
        lines = [*text, f"label: {label}" if label else "label: ____________"]
        _fit_text(fig, ax_text, ax_hist, lines, fontsize=th.fs_annot, color=COLORS["neutral"])
        rows = ([{"chips": chips, "overlays": overlays, "title": ""}] if n_chips else []) + list(extra_rows)
        all_overlays = [np.asarray(r["overlays"], float) for r in rows if r.get("overlays") is not None]
        vmax = (
            float(np.nanpercentile(np.concatenate([o.ravel() for o in all_overlays]), 98))
            if all_overlays
            else None
        )
        vmin = 0.0 if all_overlays and min(np.nanmin(o) for o in all_overlays) >= 0 else None
        drawn = []
        for j, row in enumerate(rows):
            span = 12 // max(len(row["chips"]), 1)
            axes_row = []
            for i, chip in enumerate(row["chips"]):
                ax = fig.add_subplot(gs[1 + j, i * span : (i + 1) * span])
                image(chip, rgb=True, stretch=True, ax=ax)
                if row.get("overlays") is not None:
                    overlay_grid(
                        ax,
                        row["overlays"][i],
                        window=window,
                        cmap=overlay_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        alpha=overlay_alpha,
                        mask_below=overlay_mask_below,
                    )
                if i == 0 and row.get("title"):
                    ax.set_title(row["title"], fontsize=th.fs_annot, loc="left", pad=2)
                axes_row.append(ax)
            drawn.append(axes_row)
        chip_axes = drawn[0] if n_chips else []
        extra_axes = drawn[1:] if n_chips else drawn
    return fig, {
        "map": ax_map,
        "hist": ax_hist,
        "text": ax_text,
        "chips": chip_axes,
        "extra_rows": extra_axes,
    }


def _fit_text(
    fig: Figure,
    ax_text: Axes,
    ax_hist: Axes,
    lines: Sequence[str],
    *,
    fontsize: float,
    color: str,
    min_scale: float = 0.7,
    gap_pt: float = 4.0,
) -> Text:
    """Write ``lines`` into ``ax_text`` below the histogram's labels, wrapped to the figure's right edge.

    Lines wrap at word boundaries to the width between the text axes' left edge and the figure's
    right edge; if the wrapped block still runs below the text axes, the font shrinks (down to
    ``min_scale`` times ``fontsize``).
    """
    renderer = fig.canvas.get_renderer()
    gap = gap_pt * fig.dpi / 72.0
    box = ax_text.get_window_extent(renderer)
    top = min(ax_hist.get_tightbbox(renderer).y0 - gap, box.y1)
    max_width = fig.bbox.x1 - box.x0 - gap
    artist = ax_text.text(0.0, (top - box.y0) / box.height, "", transform=ax_text.transAxes, va="top", ha="left",
                          color=color)  # fmt: skip

    def width(s: str) -> float:
        artist.set_text(s)
        return artist.get_window_extent(renderer).width

    for scale in np.linspace(1.0, min_scale, 4):
        artist.set_fontsize(fontsize * scale)
        wrapped = []
        for line in lines:
            for raw in line.split("\n"):
                n = len(raw)
                while n > 8 and any(width(seg) > max_width for seg in textwrap.wrap(raw, n) or [""]):
                    n -= max(1, n // 10)
                wrapped.extend(textwrap.wrap(raw, n) or [""])
        artist.set_text("\n".join(wrapped))
        if artist.get_window_extent(renderer).y0 >= box.y0:
            break
    return artist
