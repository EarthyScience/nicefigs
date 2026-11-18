"""Predicted-vs-observed agreement plots with density shading, 1:1 line, fit and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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


def _finite_stride(x: ArrayLike, y: ArrayLike, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite pairs of ``(x, y)``, thinned by an even stride to at most ``n`` (deterministic)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size > n:
        idx = np.linspace(0, x.size - 1, n).astype(int)
        x, y = x[idx], y[idx]
    return x, y


def kde_contours(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    color: Any,
    *,
    max_points: int = 4000,
    levels: int = 4,
    linewidth: float = 0.7,
    alpha: float = 0.9,
) -> None:
    """Overlay Gaussian-KDE density contour lines of paired samples.

    The density is evaluated over the 1st to 99th percentile of each variable, so sparse
    outliers do not drag a low contour across the panel. Too few points, no spread or a
    singular covariance draw nothing.

    Args:
        ax: Target axes.
        x: Samples; non-finite pairs are dropped.
        y: Paired samples, same length as ``x``.
        color: Contour colour.
        max_points: Pairs the density is estimated from, thinned by an even stride.
        levels: Number of contour levels.
        linewidth: Contour line width (pt).
        alpha: Contour opacity.
    """
    from scipy.stats import gaussian_kde

    x, y = _finite_stride(x, y, max_points)
    if x.size < 10 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return
    xs = np.linspace(*np.percentile(x, [1, 99]), 70)
    ys = np.linspace(*np.percentile(y, [1, 99]), 70)
    gx, gy = np.meshgrid(xs, ys)
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, z, levels=levels, colors=color, linewidths=linewidth, alpha=alpha)


def scatter_regression(
    x: ArrayLike,
    models: Mapping[str, ArrayLike],
    *,
    ax: Axes | None = None,
    colors: Mapping[str, Any],
    xlabel: str = "",
    ylabel: str = "",
    regress: Callable[[np.ndarray, np.ndarray], tuple[float, float, float, float]] | None = None,
    max_points: int = 4000,
    legend: bool = True,
) -> Axes:
    """Target-against-model panel: 1:1 line, faint scatter, density contours and a regression per model.

    Each model's legend entry carries its slope, R² and RMSE to the 1:1 line (what a model
    against a target reports is how far it is from the target, not how well a free line fits).
    Both axes span the 1st to 99th percentile of all values and are equal.

    Args:
        x: The target values.
        models: Name to values of each model against ``x``, in legend order.
        ax: Target axes; a new half-width figure if omitted.
        colors: Colour per model name.
        xlabel: x label.
        ylabel: y label.
        regress: ``(x, y) -> (slope, intercept, r2, rmse)``; ordinary least squares and the RMSE to the 1:1 line if omitted.
        max_points: Points scattered per model, thinned by an even stride.
        legend: Draw the legend.

    Returns:
        The axes.
    """
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width="half", aspect=1.0)
    th = get_theme()
    regress = regress or _regress_to_identity
    x = np.asarray(x, dtype=float)
    thinned = {name: _finite_stride(x, y, max_points) for name, y in models.items()}
    finite = np.concatenate([a for pair in thinned.values() for a in pair])
    lo, hi = np.nanpercentile(finite, [1, 99])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=1)
    for name, (xs, ys) in thinned.items():
        ax.scatter(xs, ys, s=3, alpha=0.12, color=colors[name], edgecolors="none", zorder=2)
    for name, y in models.items():
        kde_contours(ax, x, y, colors[name], max_points=max_points)
    xx = np.array([lo, hi])
    for name, y in models.items():
        slope, intercept, r2, rmse = regress(x, np.asarray(y, dtype=float))
        ax.plot(
            xx,
            slope * xx + intercept,
            color=colors[name],
            lw=1.4,
            zorder=4,
            label=f"{name}: slope {slope:.2f}, R² {r2:.2f}, RMSE {rmse:.2f}",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel, fontsize=th.fs_axis_label)
    ax.set_ylabel(ylabel, fontsize=th.fs_axis_label)
    if legend:
        ax.legend(fontsize=th.fs_legend, loc="upper left", framealpha=0.85)
    return ax


def _regress_to_identity(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``(slope, intercept, r2, rmse)`` of ``y`` on ``x``; the RMSE is to the 1:1 line."""
    ok = np.isfinite(x) & np.isfinite(y)
    a, b, r2, rmse, _ = _linfit(x[ok], y[ok])
    return float(a), float(b), float(r2), rmse
