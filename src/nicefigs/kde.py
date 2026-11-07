"""Predicted-vs-observed agreement plots with density shading, 1:1 line, fit and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .palettes import cmap as resolve_cmap
from .style import COLORS, get_theme
from .units import fmt_val


def _kde_density(ax, xi_pts, yi_pts, x, y, cm, gridsize):
    """Draw a gaussian-KDE contourf; return ``"hexbin"`` to fall back on failure.

    Robust to the degenerate cases that make ``gaussian_kde`` raise
    ``LinAlgError`` — too few points, or a (near-)singular covariance when the
    cloud is essentially collinear (e.g. plotting a variable against itself).
    """
    from scipy.stats import gaussian_kde

    if xi_pts.size <= 5:
        return "hexbin"
    try:
        kde = gaussian_kde(np.vstack([xi_pts, yi_pts]))
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        gx, gy = np.meshgrid(np.linspace(lo, hi, gridsize), np.linspace(lo, hi, gridsize))
        zi = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    except (np.linalg.LinAlgError, ValueError):
        # Singular covariance (e.g. an (almost) collinear cloud) or a degenerate
        # grid — fall back to the hexbin density instead of erroring.
        return "hexbin"
    # A (near-)collinear cloud can leave the covariance numerically singular
    # without raising, yielding non-finite or flat densities; bail to hexbin so
    # the contour levels below are always strictly increasing.
    zmax = float(np.nanmax(zi)) if np.isfinite(zi).any() else 0.0
    if not np.isfinite(zmax) or zmax <= 0:
        return "hexbin"
    # Start the fill above a small density floor so the empty corners of the
    # panel stay white instead of washing pale blue.
    levels = np.linspace(zmax * 0.05, zmax, 11)
    if not np.all(np.diff(levels) > 0):
        return "hexbin"
    ax.contourf(gx, gy, zi, levels=levels, cmap=cm, zorder=1, extend="neither")
    return ""


def _linfit(x, y):
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    return a, b, r2, rmse, bias


def agreement(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ax: Axes | None = None,
    cmap: str = "nice_teal",
    density: str = "kde",
    gridsize: int = 60,
    outlier_z: float = 3.0,
    show_outliers: bool = True,
    one_to_one: bool = True,
    regression: bool = True,
    metrics: bool = True,
    xlabel: str = "Observed",
    ylabel: str = "Predicted",
    units: str | None = None,
    equal: bool = True,
    width: float | str = "half",
    color_line: Any = None,
) -> tuple[Axes, dict[str, float]]:
    """Density agreement plot of predicted ``y`` against observed ``x``.

    Layers a density of the bulk of the cloud, a dashed 1:1 line, outliers as
    faint points, a least-squares fit line and a metrics box (n, R², RMSE, bias).
    Non-finite pairs are dropped.

    Args:
        x: Observed values, shape ``(n,)``.
        y: Predicted values, shape ``(n,)``.
        ax: Axes to draw on; a new square figure is created if ``None``.
        cmap: Colormap for the density shading.
        density: ``"kde"`` (gaussian-KDE contours, needs scipy; falls back to
            ``"hexbin"`` for degenerate clouds), ``"hexbin"`` or ``"hist"``.
        gridsize: Grid resolution of the density.
        outlier_z: Points whose residual from the 1:1 line exceeds this many
            robust standard deviations are drawn as points instead of density.
        show_outliers: Draw the outlier points.
        one_to_one: Draw the 1:1 reference line.
        regression: Draw the least-squares fit line.
        metrics: Draw the metrics box.
        xlabel: x-axis label.
        ylabel: y-axis label.
        units: Unit appended to both axis labels.
        equal: Share limits on both axes and force a square aspect.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        color_line: Colour of the fit line; defaults to the theme orange.

    Returns:
        ``(ax, stats)`` with ``stats`` keys ``slope``, ``intercept``, ``r2``,
        ``rmse`` and ``bias`` (``y - x``, in data units), and ``n``.
    """
    th = get_theme()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if ax is None:
        from .figure import subplots

        _, ax = subplots(width=width, aspect=1.0)

    a, b, r2, rmse, bias = _linfit(x, y)

    resid = y - x
    rstd = np.median(np.abs(resid - np.median(resid))) * 1.4826 or resid.std() or 1.0
    is_out = np.abs(resid - np.median(resid)) > outlier_z * rstd
    inl = ~is_out

    cm = resolve_cmap(cmap)
    if density == "kde":
        density = _kde_density(ax, x[inl], y[inl], x, y, cm, gridsize)
    if density == "hexbin":
        ax.hexbin(x[inl], y[inl], gridsize=gridsize, cmap=cm, mincnt=1, linewidths=0, zorder=1)
    elif density == "hist":
        ax.hist2d(x[inl], y[inl], bins=gridsize, cmap=cm, cmin=1, zorder=1)

    if show_outliers and is_out.any():
        ax.scatter(
            x[is_out], y[is_out], s=5, c=COLORS["neutral"], alpha=0.5, linewidths=0, zorder=2, rasterized=True
        )

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.03 * (hi - lo)
    lim = (lo - pad, hi + pad)
    if one_to_one:
        ax.plot(lim, lim, ls="--", lw=0.8, color=COLORS["neutral"], zorder=3, label="1:1")
    if regression:
        xs = np.array(lim)
        from .artists import NICE_COLORS

        ax.plot(
            xs, a * xs + b, ls="-", lw=1.2, color=color_line or NICE_COLORS["orange"], zorder=4, label="fit"
        )

    if equal:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")

    from .units import axis_label

    ax.set_xlabel(axis_label(xlabel, units))
    ax.set_ylabel(axis_label(ylabel, units))

    stats = dict(
        slope=float(a), intercept=float(b), r2=float(r2), rmse=float(rmse), bias=float(bias), n=int(x.size)
    )
    if metrics:
        txt = f"n = {stats['n']}\nR² = {fmt_val(r2)}\nRMSE = {fmt_val(rmse)}\nbias = {fmt_val(bias)}"
        ax.text(
            0.04,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=th.fs_annot,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.4, alpha=0.85),
        )
    return ax, stats
