"""Taylor diagrams: correlation, standard deviation and centred RMS difference in one polar plot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .palettes import qualitative
from .style import get_theme

#: Marker rotation for the models.
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]


def _coerce(stats):
    """Normalise input into a list of ``{name, corr, std, is_ratio}`` dicts."""
    if isinstance(stats, dict):
        out = []
        for name, v in stats.items():
            if isinstance(v, dict):
                d = dict(v)
                d.setdefault("name", name)
                out.append(d)
            else:
                std, corr = v
                out.append({"name": name, "std": std, "correlation": corr})
        stats = out
    rows = []
    for d in stats:
        corr = d.get("corr", d.get("correlation"))
        if corr is None:
            raise KeyError("each Taylor entry needs 'corr' or 'correlation'")
        if d.get("std_ratio") is not None:
            std, is_ratio = float(d["std_ratio"]), True
        elif d.get("std") is not None:
            std, is_ratio = float(d["std"]), False
        else:
            raise KeyError("each Taylor entry needs 'std' or 'std_ratio'")
        rows.append({"name": d.get("name", ""), "corr": float(corr), "std": std, "is_ratio": is_ratio})
    return rows


def taylor(
    stats: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    ax: Axes | None = None,
    ref_std: float = 1.0,
    normalize: bool = True,
    ref_label: str = "Reference",
    std_label: str | None = None,
    colors: Sequence[Any] | None = None,
    markers: Sequence[str] | None = None,
    marker_size: float = 9.0,
    rms_contours: bool = True,
    n_rms: int = 4,
    corr_ticks: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0),
    negative: bool = False,
    title: str = "",
    legend: bool = True,
    width: float | str = "half",
) -> Axes:
    """Draw a Taylor diagram of one or more models against a reference.

    Each model sits at angle ``arccos(r)`` (its correlation ``r`` with the
    reference) and radius equal to its standard deviation; its distance to the
    reference point is the centred RMS difference.

    Args:
        stats: A list of dicts, one per model, or ``{name: dict}``, or
            ``{name: (std, corr)}``. Each dict holds ``name`` (optional with the
            mapping forms), a correlation ``corr`` (or ``correlation``), and
            either an absolute ``std`` or a normalised ``std_ratio``
            (σ_model / σ_ref). Other keys are ignored.
        ax: Existing polar axes to draw on; a new one is created if ``None``.
        ref_std: Reference standard deviation, in the units of ``std``.
        normalize: Divide absolute ``std`` values by ``ref_std`` so the
            reference sits at radius 1; otherwise it sits at ``ref_std`` and
            ``std_ratio`` values are multiplied by ``ref_std``.
        ref_label: Legend label of the reference point.
        std_label: Radial-axis label; defaults to "(Normalized) standard deviation".
        colors: Per-model colours; defaults to the theme palette.
        markers: Per-model markers; defaults to :data:`MARKERS`.
        marker_size: Model marker size (pt).
        rms_contours: Draw dotted centred-RMS-difference arcs around the reference.
        n_rms: Number of RMS arcs.
        corr_ticks: Correlations marked on the angular axis.
        negative: Extend the diagram to negative correlations (half circle).
        title: Left-aligned title.
        legend: Add a legend of the model markers outside the axes.
        width: Figure width (preset name or inches) when ``ax`` is ``None``.

    Returns:
        The polar axes.

    Raises:
        KeyError: An entry has no correlation or no standard deviation.
    """
    import matplotlib.pyplot as plt

    rows = _coerce(stats)
    th = get_theme()
    ref = 1.0 if normalize else float(ref_std)

    def _std(row):
        if row["is_ratio"]:
            return row["std"] if normalize else row["std"] * ref_std
        return row["std"] / ref_std if normalize else row["std"]

    stds = [_std(r) for r in rows]

    if ax is None:
        from .figure import figure_size
        from .style import style

        with style(th):
            fig = plt.figure(figsize=figure_size(width, aspect=1.0), dpi=th.dpi)
            ax = fig.add_subplot(111, projection="polar")

    theta_max = np.pi if negative else np.pi / 2.0
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(theta_max))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    smax = max([ref, *stds]) * 1.3 if stds else ref * 1.3
    ax.set_ylim(0, smax)

    # Angular (correlation) grid.
    ticks = list(corr_ticks)
    if negative:
        ticks = sorted({*ticks, *[-t for t in corr_ticks]})
    ang = np.degrees(np.arccos(np.clip(ticks, -1, 1)))
    ax.set_thetagrids(ang, labels=[f"{t:g}" for t in ticks])
    ax.tick_params(axis="x", labelsize=th.fs_tick)
    ax.tick_params(axis="y", labelsize=th.fs_tick)

    # Centred-RMS-difference arcs, centred on the reference point (ref, 0).
    if rms_contours:
        rmax = np.hypot(smax, ref)
        for R in np.linspace(rmax / (n_rms + 1), rmax, n_rms):
            phi = np.linspace(0, 2 * np.pi, 361)
            x = ref + R * np.cos(phi)
            y = R * np.sin(phi)
            r = np.hypot(x, y)
            t = np.arctan2(y, x)
            keep = (t >= -1e-9) & (t <= theta_max + 1e-9) & (r <= smax)
            if keep.sum() > 1:
                ax.plot(t[keep], r[keep], ls=":", color="#9aa0a6", lw=0.6, zorder=1)

    # Reference point + its std arc.
    ax.plot(
        np.linspace(0, theta_max, 100),
        np.full(100, ref),
        ls="--",
        color=th.spine_color,
        lw=0.6,
        alpha=0.6,
        zorder=1,
    )
    ax.plot(
        0,
        ref,
        marker="*",
        markersize=marker_size + 5,
        color=th.spine_color,
        zorder=6,
        label=ref_label,
        linestyle="none",
    )

    palette = colors or qualitative(len(rows))
    marks = markers or MARKERS
    for i, row in enumerate(rows):
        ax.plot(
            np.arccos(np.clip(row["corr"], -1, 1)),
            _std(row),
            marker=marks[i % len(marks)],
            color=palette[i % len(palette)],
            markersize=marker_size,
            linestyle="none",
            zorder=5,
            label=row["name"] or None,
        )

    ax.set_rlabel_position(90 if negative else 135)
    lbl = std_label or ("Normalized standard deviation" if normalize else "Standard deviation")
    ax.set_ylabel(lbl, fontsize=th.fs_axis_label, labelpad=22)
    ax.text(
        np.radians(45 if not negative else 90),
        smax * 1.12,
        "Correlation",
        ha="center",
        va="center",
        fontsize=th.fs_axis_label,
        rotation=(-45 if not negative else 0),
    )
    if title:
        ax.set_title(title, fontsize=th.fs_title, loc="left", pad=10)
    if legend and any(r["name"] for r in rows):
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.32, 1.08),
            fontsize=th.fs_legend,
            framealpha=0.9,
            markerscale=0.6,
        )
    return ax
