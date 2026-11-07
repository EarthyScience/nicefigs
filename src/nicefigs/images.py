"""Images, heatmaps and RGB composites, singly or in grids with shared colorbars."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .colorbars import colorbar
from .figure import subplots
from .palettes import cmap as resolve_cmap
from .palettes import robust_limits
from .style import get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

#: Basis returned by :func:`pca_rgb`: ``(mean, components, limits)``.
PCABasis = tuple[np.ndarray, np.ndarray, np.ndarray]


def _stretch_rgb(img: ArrayLike, *, percentile: tuple[float, float] = (2, 98)) -> np.ndarray:
    """Per-channel percentile contrast stretch to [0, 1] for display."""
    img = np.asarray(img, dtype=float)
    if img.max() > 1.5:  # looks like 0..255
        img = img / 255.0
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        lo, hi = np.nanpercentile(img[..., c], percentile)
        out[..., c] = np.clip((img[..., c] - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def image(
    data: ArrayLike,
    *,
    ax: Axes | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    robust: bool = False,
    rgb: bool = False,
    stretch: bool = False,
    title: str = "",
    interpolation: str = "nearest",
    origin: Literal["upper", "lower"] = "upper",
    overlay_contour: ArrayLike | None = None,
    contour_color: str = "white",
    contour_linewidth: float = 0.8,
    frame: bool = False,
    **imshow_kw: Any,
) -> tuple[Axes, AxesImage | None]:
    """Show an intensity or RGB array on ``ax`` without ticks.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        data: Intensity array ``(H, W)``, or colour image ``(H, W, 3|4)`` in
            ``[0, 1]`` or ``[0, 255]``. A 3-D array with 3 or 4 channels is always
            shown as RGB.
        cmap: Colormap name for intensity data.
        vmin: Lower colour limit for intensity data.
        vmax: Upper colour limit for intensity data.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        robust: Use the 2nd/98th percentiles when no limits or ``norm`` are given.
        rgb: Treat ``data`` as a colour image.
        stretch: Contrast-stretch each RGB channel to its 2nd/98th percentiles.
        title: Axes title.
        interpolation: Passed to :meth:`~matplotlib.axes.Axes.imshow`.
        origin: Passed to :meth:`~matplotlib.axes.Axes.imshow`.
        overlay_contour: Mask of shape ``(H, W)``; its 0.5 contour is drawn on top.
        contour_color: Colour of the overlay contour.
        contour_linewidth: Line width of the overlay contour in points.
        frame: Draw a thin box on all four sides; otherwise no spines.
        **imshow_kw: Forwarded to :meth:`~matplotlib.axes.Axes.imshow`.

    Returns:
        ``(ax, image)``: the axes and the :class:`~matplotlib.image.AxesImage` of
        intensity data (for a colorbar), or ``None`` for RGB data.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    if rgb or (np.ndim(data) == 3 and np.asarray(data).shape[-1] in (3, 4)):
        img = (
            _stretch_rgb(data)
            if stretch
            else np.clip(np.asarray(data, float) / (255.0 if np.nanmax(data) > 1.5 else 1.0), 0, 1)
        )
        ax.imshow(img, interpolation=interpolation, origin=origin, aspect="equal", **imshow_kw)
        mappable = None
    else:
        if robust and vmin is None and vmax is None and norm is None:
            vmin, vmax = robust_limits(data)
        kw = dict(cmap=resolve_cmap(cmap), interpolation=interpolation, origin=origin, aspect="equal")
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"], kw["vmax"] = vmin, vmax
        mappable = ax.imshow(data, **kw, **imshow_kw)
    if overlay_contour is not None:
        ax.contour(
            np.asarray(overlay_contour, dtype=float),
            levels=[0.5],
            colors=contour_color,
            linewidths=contour_linewidth,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    # Symmetric spines: a thin box on all four sides, or none at all — never the
    # theme's asymmetric top/right-off look, which reads oddly around an image.
    for sp in ax.spines.values():
        sp.set_visible(frame)
        if frame:
            sp.set_linewidth(0.5)
            sp.set_color("#888888")
    if title:
        ax.set_title(title, fontsize=th.fs_title, pad=2.5)
    return ax, mappable


def overlay_grid(
    ax: Axes,
    values: ArrayLike,
    *,
    window: tuple[int, int, int],
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.6,
    mask_below: float | None = None,
    outline: bool = True,
    outline_color: str = "white",
    outline_linewidth: float = 0.6,
) -> AxesImage:
    """Overlay a coarse ``(h, w)`` grid of values on a square window of an image.

    The grid is stretched over image pixels ``[row, row + size)`` x ``[col, col + size)``
    of the image already on ``ax`` (pixel-centre coordinates, as :func:`image` draws it),
    so grid cell ``(i, j)`` covers ``size / h`` x ``size / w`` image pixels. The axes
    limits are left as they were.

    Args:
        ax: Axes holding the image.
        values: Grid of shape ``(h, w)``; NaN cells are transparent.
        window: ``(row, col, size)`` of the window, in image pixels.
        cmap: Colormap name of the grid.
        vmin: Lower colour limit of the grid.
        vmax: Upper colour limit of the grid.
        alpha: Opacity of the grid.
        mask_below: Cells with values at or below this are transparent.
        outline: Draw the window's frame.
        outline_color: Frame colour.
        outline_linewidth: Frame line width in points.

    Returns:
        The grid's ``AxesImage``.
    """
    import matplotlib.patches as mpatches

    row, col, size = window
    grid = np.ma.masked_invalid(np.asarray(values, dtype=float))
    if mask_below is not None:
        grid = np.ma.masked_where(grid <= mask_below, grid)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    extent = (col - 0.5, col + size - 0.5, row + size - 0.5, row - 0.5)
    im = ax.imshow(
        grid,
        cmap=resolve_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        extent=extent,
        interpolation="nearest",
        zorder=3,
    )
    if outline:
        ax.add_patch(
            mpatches.Rectangle(
                (col - 0.5, row - 0.5),
                size,
                size,
                fill=False,
                edgecolor=outline_color,
                linewidth=outline_linewidth,
                zorder=4,
            )
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def pca_rgb(
    field: ArrayLike,
    *,
    basis: PCABasis | tuple[np.ndarray, np.ndarray] | None = None,
    percentile: tuple[float, float] = (2, 98),
    return_basis: bool = False,
) -> np.ndarray | tuple[np.ndarray, PCABasis]:
    """Map a multi-channel field to RGB through its first three principal components.

    Each component is scaled to ``[0, 1]`` between its ``percentile`` limits and
    mapped to R, G, B.

    Args:
        field: Array of shape ``(..., D)`` with ``D >= 3``, e.g. ``(H, W, D)``.
        basis: Reuse a projection. A ``(mean, components, limits)`` basis from
            ``return_basis=True`` colours several fields identically; a
            ``(mean, components)`` pair reuses the projection but rescales each
            field to its own percentiles. ``None`` fits the PCA on ``field``.
        percentile: Lower/upper percentiles mapped to 0 and 1.
        return_basis: Also return the ``(mean, components, limits)`` basis.

    Returns:
        RGB array of shape ``field.shape[:-1] + (3,)``, or ``(rgb, basis)`` when
        ``return_basis``.
    """
    arr = np.asarray(field, dtype=float)
    lead = arr.shape[:-1]
    flat = arr.reshape(-1, arr.shape[-1])
    limits = None
    if basis is None:
        mean = flat.mean(0)
        comps = np.linalg.svd(flat - mean, full_matrices=False)[2][:3]
    elif len(basis) == 2:
        mean, comps = basis
    else:
        mean, comps, limits = basis[0], basis[1], np.asarray(basis[2], dtype=float)
    proj = (flat - mean) @ comps.T  # (N, 3)
    if limits is None:
        limits = np.stack([np.nanpercentile(proj[:, c], percentile) for c in range(3)])
    out = np.clip((proj - limits[:, 0]) / np.maximum(limits[:, 1] - limits[:, 0], 1e-9), 0, 1)
    rgb = out.reshape(*lead, 3)
    return (rgb, (mean, comps, limits)) if return_basis else rgb


def heatmap(
    matrix: ArrayLike,
    *,
    ax: Axes | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    cmap: str = "Spectral_r",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    annotate: bool = False,
    annot_fmt: str = "{:.2f}",
    annot_size: float | None = None,
    grid: bool = True,
    title: str = "",
) -> tuple[Axes, AxesImage]:
    """Draw a labelled heatmap of a 2-D matrix, optionally annotating each cell.

    Args:
        ax: Axes to draw on; a new figure is created if ``None``.
        matrix: Values of shape ``(rows, cols)``; NaN cells are left blank.
        row_labels: Tick labels for the rows.
        col_labels: Tick labels for the columns, rotated 45°.
        cmap: Colormap name.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        annotate: Write each value in its cell, in black or white by cell luminance.
        annot_fmt: :meth:`str.format` pattern for annotations.
        annot_size: Annotation size in points; defaults to the theme's annotation size.
        grid: Separate cells with thin white lines.
        title: Axes title.

    Returns:
        ``(ax, image)``: the axes and the drawn :class:`~matplotlib.image.AxesImage`.
    """
    if ax is None:
        _, ax = subplots()
    th = get_theme()
    matrix = np.asarray(matrix, dtype=float)
    kw = dict(cmap=resolve_cmap(cmap), aspect="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    im = ax.imshow(matrix, **kw)
    nr, nc = matrix.shape
    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nr))
    if col_labels is not None:
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=th.fs_tick)
    if row_labels is not None:
        ax.set_yticklabels(row_labels, fontsize=th.fs_tick)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if grid:
        ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
    if annotate:
        fs = annot_size or th.fs_annot
        for i in range(nr):
            for j in range(nc):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(
                    j,
                    i,
                    annot_fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color="white" if lum < 0.55 else "#222222",
                )
    if title:
        ax.set_title(title, fontsize=th.fs_title, pad=3)
    return ax, im


def image_grid(
    images: Sequence[ArrayLike],
    *,
    nrows: int | None = None,
    ncols: int | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    robust: bool = True,
    rgb: bool = False,
    stretch: bool = False,
    shared_scale: bool = True,
    colorbar_mode: Literal["single", "row", "each", "none"] = "single",
    colorbar_label: str = "",
    titles: Sequence[str] | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    width: str | float = "full",
    panel_aspect: float = 1.0,
    wspace: float = 0.06,
    hspace: float = 0.1,
    fig: Figure | None = None,
) -> tuple[Figure, np.ndarray]:
    """Lay out images in a grid, with one, per-row, per-image or no colorbars.

    Args:
        images: 2-D intensity or 3-D RGB arrays, filled row by row.
        nrows: Number of rows; inferred (near-square grid) if omitted.
        ncols: Number of columns; inferred if omitted.
        cmap: Colormap name for intensity images.
        vmin: Lower colour limit; overrides the shared scale.
        vmax: Upper colour limit; overrides the shared scale.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        robust: Base the shared scale on the 2nd/98th percentiles, not min/max.
        rgb: Treat every image as a colour image.
        stretch: Contrast-stretch RGB images, as in :func:`image`.
        shared_scale: Use one colour range across all intensity images.
        colorbar_mode: ``"single"`` (one bar for the grid), ``"row"`` (one per
            row), ``"each"`` (one per image) or ``"none"``.
        colorbar_label: Label of every colorbar.
        titles: Per-image titles.
        row_labels: Labels left of the first column.
        col_labels: Labels above the first row.
        width: Figure width, a named width or inches (see :func:`nicefigs.figure_size`).
        panel_aspect: Height / width of each panel.
        wspace: Horizontal gap between panels, as a fraction of panel width.
        hspace: Vertical gap between panels, as a fraction of panel height.
        fig: Existing figure to draw into; a new one is created if ``None``.

    Returns:
        ``(fig, axes)`` with ``axes`` a ``(nrows, ncols)`` object array; cells
        beyond ``len(images)`` are switched off.
    """
    from .figure import figure_size
    from .style import style

    th = get_theme()
    n = len(images)
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))

    if shared_scale and not rgb and norm is None and vmin is None and vmax is None:
        allv = np.concatenate([np.asarray(im, float).ravel() for im in images])
        vmin, vmax = robust_limits(allv) if robust else (float(np.nanmin(allv)), float(np.nanmax(allv)))

    size = figure_size(width, panel_aspect=panel_aspect, nrows=nrows, ncols=ncols)
    with style(th):
        if fig is None:
            fig = plt.figure(figsize=size, dpi=th.dpi)
        axes = np.empty((nrows, ncols), dtype=object)
        mappables = np.empty((nrows, ncols), dtype=object)
        gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace)
        for k in range(nrows * ncols):
            i, j = divmod(k, ncols)
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            if k < n:
                _, m = image(
                    images[k],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    norm=norm,
                    rgb=rgb,
                    stretch=stretch,
                    title=(titles[k] if titles is not None and k < len(titles) else ""),
                    ax=ax,
                )
                mappables[i, j] = m
                if colorbar_mode == "each" and m is not None:
                    colorbar(m, ax, label=colorbar_label, thickness=0.02)
            else:
                ax.axis("off")
            if row_labels is not None and j == 0 and i < len(row_labels):
                ax.set_ylabel(row_labels[i], fontsize=th.fs_axis_label)
            if col_labels is not None and i == 0 and j < len(col_labels):
                ax.set_title(col_labels[j], fontsize=th.fs_axis_label, pad=3)

        valid = [m for m in mappables.ravel() if m is not None]
        if colorbar_mode == "single" and valid:
            colorbar(valid[0], list(axes.ravel()), label=colorbar_label, location="right")
        elif colorbar_mode == "row":
            for i in range(nrows):
                row_m = [m for m in mappables[i] if m is not None]
                if row_m:
                    colorbar(row_m[0], list(axes[i]), label=colorbar_label, location="right")
    return fig, axes
