"""Eigen-spectrum plots: eigenvalue decay against rank, or cumulative explained variance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .lines import lines
from .palettes import qualitative, role_style
from .panels import reference_line

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

#: Colour of the threshold guides of a cumulative spectrum.
_GUIDE_COLOR = "#888888"


def _as_named_spectra(
    eigenvalues: ArrayLike | Sequence[ArrayLike] | Mapping[str, ArrayLike],
    labels: Sequence[str] | None,
    rtol: float,
) -> dict[str, np.ndarray]:
    """Return ``{name: eigenvalues above rtol * max, sorted descending}``."""
    if isinstance(eigenvalues, Mapping):
        named = dict(eigenvalues)
    elif all(np.ndim(v) == 0 for v in eigenvalues):
        named = {"spectrum": eigenvalues}
    else:
        named = {str(i): ev for i, ev in enumerate(eigenvalues)}
    if labels is not None:
        if len(labels) != len(named):
            raise ValueError(f"got {len(labels)} labels for {len(named)} spectra")
        named = dict(zip(labels, named.values()))
    out = {}
    for name, ev in named.items():
        ev = np.asarray(ev, dtype=float)
        if ev.ndim != 1:
            raise ValueError(f"spectrum {name!r} must be 1-D, got shape {ev.shape}")
        ev = np.sort(ev[np.isfinite(ev) & (ev > 0)])[::-1]
        if ev.size:
            ev = ev[ev > rtol * ev[0]]
        if ev.size == 0:
            raise ValueError(f"spectrum {name!r} has no positive eigenvalues")
        out[name] = ev
    return out


def spectrum(
    eigenvalues: ArrayLike | Sequence[ArrayLike] | Mapping[str, ArrayLike],
    *,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    colors: Mapping[str, Any] | Sequence[Any] | None = None,
    styles: Mapping[str, str] | None = None,
    normalize: bool = True,
    rtol: float = 1e-10,
    cumulative: bool = False,
    thresholds: Sequence[float] = (0.5, 0.9, 0.99),
    reference_alpha: float | None = None,
    xlabel: str = "rank $k$",
    ylabel: str | None = None,
    title: str = "",
    legend: bool = True,
    width: float | str = "half",
    aspect: float = 0.72,
) -> Axes:
    """Plot eigenvalue spectra against rank ``k``, counted from 1.

    By default this draws the log-log decay of ``λ_k``, or of ``λ_k / Σλ`` when
    ``normalize``. With ``cumulative=True`` it draws the cumulative
    explained-variance fraction on a log rank axis instead, with one labelled
    horizontal guide per threshold and, for each series, a marker where it
    first reaches that threshold. Each spectrum is sorted descending, and its
    non-finite, non-positive and round-off-sized (``λ ≤ rtol · λ_max``)
    eigenvalues are dropped.

    Args:
        eigenvalues: One 1-D array, a sequence of 1-D arrays, or ``{name: array}``.
        ax: Axes to draw on; a new figure is created if ``None``.
        labels: Series names for array or sequence input; overrides dict keys.
        colors: ``{name: colour}`` or a list in series order; defaults to
            :func:`~nicefigs.qualitative`.
        styles: ``{name: linestyle}``.
        normalize: Plot ``λ_k / Σλ`` instead of ``λ_k``; ignored when ``cumulative``.
        rtol: Relative floor below which eigenvalues count as numerical zeros;
            ``0`` keeps every positive eigenvalue.
        cumulative: Plot ``Σ_{j≤k} λ_j / Σλ`` instead of the spectrum.
        thresholds: Explained-variance fractions marked when ``cumulative``.
        reference_alpha: Draw a ``k^-α`` power law through the plotted leading
            value of the first series; spectrum mode only.
        xlabel: x-axis label.
        ylabel: y-axis label; defaults to the plotted quantity.
        title: Panel title.
        legend: Add a legend when more than one line is labelled.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.
        aspect: Figure ``height / width`` when ``ax`` is ``None``.

    Returns:
        The axes.

    Raises:
        ValueError: ``reference_alpha`` is given with ``cumulative``, a spectrum
            is not 1-D or has no positive eigenvalue, or ``labels`` has the
            wrong length.
    """
    if cumulative and reference_alpha is not None:
        raise ValueError("reference_alpha applies to the spectrum, not cumulative=True")
    spectra = _as_named_spectra(eigenvalues, labels, rtol)
    names = list(spectra)
    if colors is None:
        colors = dict(zip(names, qualitative(len(names))))
    elif not isinstance(colors, Mapping):
        colors = dict(zip(names, colors))

    if cumulative:
        series = {n: np.cumsum(ev) / ev.sum() for n, ev in spectra.items()}
        default_ylabel = "cumulative explained variance"
    elif normalize:
        series = {n: ev / ev.sum() for n, ev in spectra.items()}
        default_ylabel = r"$\lambda_k / \sum \lambda$"
    else:
        series = spectra
        default_ylabel = r"$\lambda_k$"
    ranks = {n: np.arange(1, len(y) + 1) for n, y in series.items()}

    ax = lines(
        ranks,
        series,
        ax=ax,
        colors=colors,
        styles=styles,
        logx=True,
        logy=not cumulative,
        xlabel=xlabel,
        ylabel=default_ylabel if ylabel is None else ylabel,
        title=title,
        legend=False,
        width=width,
        aspect=aspect,
    )

    if cumulative:
        _mark_thresholds(ax, ranks, series, colors, thresholds)
        ax.set_ylim(0, 1.02)
    elif reference_alpha is not None:
        k = np.geomspace(1, max(len(r) for r in ranks.values()), 50)
        lead = series[names[0]][0]  # the plotted value at k = 1, normalised or not
        ax.plot(
            k,
            lead * k ** -float(reference_alpha),
            label=f"$k^{{-{reference_alpha:g}}}$",
            **role_style("reference", kind="line"),
        )

    if legend and len([lbl for lbl in ax.get_legend_handles_labels()[1] if lbl]) > 1:
        ax.legend(loc="best")
    return ax


def _mark_thresholds(
    ax: Axes,
    ranks: Mapping[str, np.ndarray],
    series: Mapping[str, np.ndarray],
    colors: Mapping[str, Any],
    thresholds: Sequence[float],
) -> None:
    """Draw one labelled guide per threshold and, per series, a marker at the first rank reaching it."""
    for t in thresholds:
        reference_line(ax, t, axis="y", label=f"{100 * t:g}%", color=_GUIDE_COLOR, linestyle=":")
    for name, y in series.items():
        # A crossing exact in real arithmetic must not be lost to rounding.
        reached = [t for t in thresholds if y[-1] >= t - 1e-12]
        idx = [int(np.argmax(y >= t - 1e-12)) for t in reached]
        ax.plot(
            ranks[name][idx],
            y[idx],
            linestyle="None",
            marker="o",
            markersize=3.5,
            color=colors[name],
            label=f"_{name} thresholds",
            zorder=4,
        )
