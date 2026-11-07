"""Model-comparison figures: one panel per metric, one violin, box or strip per model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .palettes import qualitative
from .style import get_theme


def _as_grouped(metric_scores):
    """Normalise ``{model: arr}`` or ``{model: {group: arr}}`` → ``{model:{group:arr}}``.

    Returns ``(normalised, groups_or_None)``.
    """
    first = next(iter(metric_scores.values()))
    if isinstance(first, dict):
        groups = list(dict.fromkeys(g for m in metric_scores.values() for g in m))
        return metric_scores, groups
    return {m: {"": v} for m, v in metric_scores.items()}, None


def _violin(ax, data, pos, color, width, kind):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    if kind == "violin":
        parts = ax.violinplot([data], positions=[pos], widths=width, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor("none")
            b.set_alpha(0.65)
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color="#333333", lw=1.4, zorder=4, solid_capstyle="round")
        ax.scatter([pos], [med], s=6, color="white", edgecolor="#333333", linewidth=0.5, zorder=5)
    elif kind == "box":
        bp = ax.boxplot(
            [data],
            positions=[pos],
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#333333", lw=1.0),
            whiskerprops=dict(color="#666666", lw=0.7),
            capprops=dict(color="#666666", lw=0.7),
            boxprops=dict(edgecolor="none"),
        )
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.7)
    elif kind == "strip":
        jitter = (np.random.default_rng(0).uniform(-1, 1, data.size)) * width * 0.3
        ax.scatter(pos + jitter, data, s=5, color=color, alpha=0.5, linewidths=0)
        ax.plot([pos - width / 3, pos + width / 3], [np.median(data)] * 2, color="#333333", lw=1.2)


def model_comparison(
    scores: Mapping[str, Any],
    *,
    models: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    group_label: str = "",
    kind: str = "violin",
    ncols: int | None = None,
    metric_units: Mapping[str, str] | None = None,
    metric_lims: Mapping[str, tuple[float, float]] | None = None,
    model_colors: Mapping[str, Any] | None = None,
    reference: float | Mapping[str, float] | None = None,
    reference_label: str = "",
    median_labels: bool = False,
    divider_after: int | None = None,
    width: float | str = "full",
    panel_aspect: float = 0.85,
    group_gap: float = 0.82,
    legend: bool = True,
    sharey: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Grid of metric panels comparing the score distributions of several models.

    Args:
        scores: ``{metric: {model: values}}``, or ``{metric: {model: {group: values}}}``
            to dodge a second factor (e.g. dataset) within each model. Also accepts
            the dict returned by :func:`nicefigs.data.model_scores`.
        models: Model order on the x-axis; inferred if ``None``.
        metrics: Metric (panel) order; inferred if ``None``.
        groups: Group order within each model; inferred if ``None``.
        group_label: Title of the shared group legend.
        kind: ``"violin"``, ``"box"`` or ``"strip"``.
        ncols: Panel columns; defaults to one row for up to three metrics.
        metric_units: ``{metric: unit}`` shown as the y-axis label.
        metric_lims: ``{metric: (lo, hi)}`` y-limits.
        model_colors: ``{model: colour}``; colours by model instead of by group.
        reference: Reference line value, one for all panels or ``{metric: value}``.
        reference_label: Label of the reference line; defaults to its value.
        median_labels: Annotate each distribution with its median.
        divider_after: Draw a vertical divider after this many models.
        width: Figure width (preset name or inches).
        panel_aspect: Height / width of each panel.
        group_gap: Fraction of the slot per model shared by its groups.
        legend: Add the shared group legend.
        sharey: Share the y-axis across panels.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat array, one per metric; unused grid
        cells are hidden.

    Raises:
        ValueError: ``kind`` is not ``"violin"``, ``"box"`` or ``"strip"``.
    """
    if kind not in ("violin", "box", "strip"):
        raise ValueError(f"kind must be 'violin', 'box' or 'strip', got {kind!r}")
    th = get_theme()
    if isinstance(scores, dict) and "scores" in scores and isinstance(scores["scores"], dict):
        meta = scores
        scores = meta["scores"]
        models = models or meta.get("models")
        metrics = metrics or meta.get("metrics")
        groups = groups or meta.get("timescales")

    metrics = metrics or list(scores)
    first_metric = scores[metrics[0]]
    models = models or list(first_metric)

    _, inferred_groups = _as_grouped(first_metric)
    groups = groups if groups is not None else inferred_groups

    n = len(metrics)
    ncols = ncols or (n if n <= 3 else int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    from .figure import figure_size
    from .style import style

    size = figure_size(width, nrows=nrows, ncols=ncols, panel_aspect=panel_aspect)
    with style(th):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, dpi=th.dpi, squeeze=False, sharey=sharey)
    axes_flat = axes.ravel()

    gcolors = dict(zip(groups, qualitative(len(groups)))) if groups else None
    ng = len(groups) if groups else 1

    for mi, metric in enumerate(metrics):
        ax = axes_flat[mi]
        norm, _ = _as_grouped(scores[metric])
        base = np.arange(len(models))
        for gi, g in enumerate(groups or [""]):
            offset = (gi - (ng - 1) / 2) * (group_gap / ng)
            wdt = (group_gap / ng) * 0.9
            for mj, model in enumerate(models):
                color = (
                    (model_colors or {}).get(model)
                    if model_colors
                    else (gcolors[g] if gcolors else th.palette[0])
                )
                d = norm.get(model, {}).get(g, [])
                _violin(ax, d, base[mj] + offset, color, wdt, kind)
                if median_labels and np.size(d):
                    med = float(np.nanmedian(np.asarray(d, dtype=float)))
                    ax.annotate(
                        f"{med:.2g}",
                        (base[mj] + offset, med),
                        fontsize=th.fs_annot - 0.5,
                        ha="center",
                        va="bottom",
                        xytext=(0, 2),
                        textcoords="offset points",
                    )
        ref = reference.get(metric) if isinstance(reference, dict) else reference
        if ref is not None:
            from .panels import reference_line

            reference_line(
                ax, ref, axis="y", label=reference_label, fmt="{label}" if reference_label else "{value:.2g}"
            )
        if divider_after is not None and 0 < divider_after < len(models):
            ax.axvline(divider_after - 0.5, color="#bbbbbb", linewidth=0.7, linestyle="-", zorder=0)
        ax.set_xticks(base)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=th.fs_tick)
        from .units import prettify_unit

        unit = (metric_units or {}).get(metric)
        ax.set_ylabel(prettify_unit(unit) if unit else "", fontsize=th.fs_axis_label)
        ax.set_title(metric, fontsize=th.fs_title, loc="left", pad=3)
        if metric_lims and metric in metric_lims:
            ax.set_ylim(*metric_lims[metric])
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)

    for k in range(n, len(axes_flat)):
        axes_flat[k].axis("off")

    top = 1.0
    if legend and groups:
        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(facecolor=gcolors[g], edgecolor="none", alpha=0.7, label=str(g)) for g in groups
        ]
        leg = fig.legend(
            handles=handles,
            title=group_label or None,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(groups),
            fontsize=th.fs_legend,
            title_fontsize=th.fs_legend,
            frameon=False,
        )
        # Reserve exactly the legend's height so it never overlaps the panel titles.
        bbox = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        top = bbox.y0 - 0.01
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig, axes_flat
