"""Ternary encodings: three-part compositions drawn as barycentric blends of corner colours.

A composition ``(a, b, c)`` with ``a + b + c = 1`` (sand/silt/clay, three land-cover
fractions, ...) becomes one colour via :func:`color`. The same encoding is used by
:func:`ternary_scatter`, :func:`ternary_map` and :func:`ternary_image`, and
:func:`ternary_legend` draws the triangular key that decodes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from numpy.typing import ArrayLike

from .style import get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure

#: A ternary palette name (key of :data:`TERNARY_PALETTES`) or three colours.
Corners = str | Sequence[str]

# Triangle vertices in axes coords: component 0 → bottom-left, 1 → bottom-right,
# 2 → top. Colours are colour-blind-legible by default.
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

#: Named ternary corner palettes (component 0, 1, 2 -> colour). ``"bright"`` is the
#: Paul Tol *bright* blue / red / green.
TERNARY_PALETTES: dict[str, tuple[str, str, str]] = {
    "default": ("#d73027", "#1a9850", "#4575b4"),  # red / green / blue (ColorBrewer)
    "bright": ("#4477AA", "#EE6677", "#228833"),  # Paul Tol bright: blue / red / green
    "muted": ("#332288", "#CC6677", "#117733"),  # Paul Tol muted
    "warm": ("#d73027", "#fc8d59", "#fee090"),
    # Light CMY-like corners that keep mixtures bright.
    "prevalence": ("#FFEB33", "#EB4DC7", "#66C7FF"),  # yellow / magenta / light-blue
}
DEFAULT_CORNERS = TERNARY_PALETTES["default"]


def resolve_corners(corners: Corners) -> tuple[str, str, str]:
    """Resolve a palette name or three explicit colours to a 3-tuple of colours.

    Raises:
        KeyError: ``corners`` is a name not in :data:`TERNARY_PALETTES`.
        ValueError: ``corners`` is a sequence that is not of length 3.
    """
    if isinstance(corners, str):
        try:
            return TERNARY_PALETTES[corners]
        except KeyError as exc:
            raise KeyError(f"Unknown ternary palette {corners!r}. Known: {sorted(TERNARY_PALETTES)}") from exc
    corners = tuple(corners)
    if len(corners) != 3:
        raise ValueError(f"ternary corners must be a name or 3 colours, got {corners!r}")
    return corners


def normalize_composition(comp: ArrayLike) -> np.ndarray:
    """Rescale compositions of shape ``(..., 3)`` so each sums to 1; all-zero rows become NaN."""
    c = np.asarray(comp, dtype=float)
    s = c.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s > 0, c / s, np.nan)
    return out


def color(
    comp: ArrayLike, *, corners: Corners = DEFAULT_CORNERS, gamma: float = 0.8, brighten: float = 1.0
) -> np.ndarray:
    """Colour each composition by barycentric blending of three corner colours.

    Args:
        comp: Compositions of shape ``(..., 3)``, e.g. ``(N, 3)`` or ``(H, W, 3)``;
            normalised to sum to 1.
        corners: Palette name or three colours for components 0, 1, 2.
        gamma: Exponent applied to the blended RGB; values below 1 brighten mixtures.
        brighten: Multiplicative factor applied before ``gamma``, then clipped to 1.

    Returns:
        RGB array of shape ``comp.shape`` (trailing axis of 3) in ``[0, 1]``.
    """
    w = normalize_composition(comp)
    cols = np.array([to_rgb(c) for c in resolve_corners(corners)])  # (3,3)
    rgb = np.tensordot(w, cols, axes=([-1], [0]))  # (..., 3)
    rgb = np.clip(rgb * brighten, 0, 1) ** gamma
    return rgb


def barycentric_to_xy(comp: ArrayLike) -> np.ndarray:
    """Map compositions of shape ``(..., 3)`` to ``(x, y)`` positions in the unit-edge triangle.

    Component 0 sits at ``(0, 0)``, 1 at ``(1, 0)`` and 2 at the apex ``(0.5, sqrt(3)/2)``.
    """
    w = normalize_composition(comp)
    return w @ _V


def ternary_axes(
    ax: Axes | None = None,
    *,
    labels: Sequence[str] = ("A", "B", "C"),
    gridlines: int = 5,
    corner_colors: Sequence[str] | None = None,
    fontsize: float | None = None,
) -> Axes:
    """Draw a ternary triangle frame (outline, gridlines, corner labels) on ``ax``.

    Args:
        ax: Axes to draw into; a new figure is created if ``None``.
        labels: Corner labels for components 0, 1, 2.
        gridlines: Number of intervals per edge; ``gridlines - 1`` interior lines
            are drawn parallel to each edge.
        corner_colors: Label colours per corner; dark grey if ``None``.
        fontsize: Label size in points; defaults to the theme's axis-label size.

    Returns:
        The axes, with limits, equal aspect and hidden ticks/spines set.
    """
    th = get_theme()
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width="half", aspect=np.sqrt(3) / 2 + 0.12)
    fs = fontsize or th.fs_axis_label
    tri = np.vstack([_V, _V[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#333333", lw=0.8, zorder=2)
    for k in range(1, gridlines):
        f = k / gridlines
        # lines parallel to each edge
        for i, j, m in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            p1 = _V[i] * (1 - f) + _V[m] * f
            p2 = _V[j] * (1 - f) + _V[m] * f
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#cccccc", lw=0.35, zorder=1)
    ccols = corner_colors or ["#333333"] * 3
    offs = [(-0.02, -0.04, "right", "top"), (0.02, -0.04, "left", "top"), (0.0, 0.03, "center", "bottom")]
    for v, lab, cc, (dx, dy, ha, va) in zip(_V, labels, ccols, offs):
        ax.text(v[0] + dx, v[1] + dy, lab, ha=ha, va=va, fontsize=fs, color=cc, fontweight="bold")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_aspect("equal")
    # Hide the frame but keep the background patch (so insets can be given a
    # white backing) — ``axis("off")`` would suppress the patch too.
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def ternary_scatter(
    comp: ArrayLike,
    values: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    corners: Corners = DEFAULT_CORNERS,
    labels: Sequence[str] = ("A", "B", "C"),
    size: float = 10,
    cmap: str = "inferno",
    color_by: Literal["composition", "value"] = "composition",
    **kw: Any,
) -> tuple[Axes, PathCollection | None]:
    """Scatter compositions inside a ternary triangle.

    Args:
        comp: Compositions of shape ``(N, 3)``.
        values: Per-point values of shape ``(N,)``, used when ``color_by="value"``.
        ax: Axes to draw into; a new figure is created if ``None``.
        corners: Palette name or three corner colours.
        labels: Corner labels for components 0, 1, 2.
        size: Marker area in points².
        cmap: Colormap name for ``color_by="value"``.
        color_by: ``"composition"`` colours points by their ternary colour;
            ``"value"`` colours them by ``values`` through ``cmap``.
        **kw: Forwarded to :meth:`matplotlib.axes.Axes.scatter`.

    Returns:
        ``(ax, mappable)``; ``mappable`` is the scatter collection when colouring by
        value (for a colorbar) and ``None`` otherwise.
    """
    ax = ternary_axes(ax, labels=labels, corner_colors=list(resolve_corners(corners)))
    xy = barycentric_to_xy(comp)
    if color_by == "value" and values is not None:
        from .palettes import cmap as resolve_cmap

        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=np.asarray(values), cmap=resolve_cmap(cmap), s=size, linewidths=0, **kw
        )
        return ax, sc
    cols = color(comp, corners=corners)
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=size, linewidths=0, **kw)
    return ax, None


def ternary_legend(
    ax: Axes | None = None,
    *,
    corners: Corners = DEFAULT_CORNERS,
    labels: Sequence[str] = ("A", "B", "C"),
    n: int = 160,
    gamma: float = 0.8,
    points: ArrayLike | None = None,
    point_kw: dict[str, Any] | None = None,
    fontsize: float | None = None,
) -> Axes:
    """Draw the filled colour triangle that decodes a ternary encoding.

    Args:
        ax: Axes to draw into, typically ``parent.inset_axes([...])``; a new
            figure is created if ``None``.
        corners: Palette name or three corner colours.
        labels: Corner labels for components 0, 1, 2.
        n: Raster resolution of the triangle per side.
        gamma: As in :func:`color`.
        points: Optional compositions ``(N, 3)`` overlaid as dots to show where the
            data lie.
        point_kw: Overrides for the overlay's :meth:`~matplotlib.axes.Axes.scatter` keywords.
        fontsize: Corner-label size in points.

    Returns:
        The legend axes.
    """
    ax = ternary_axes(ax, labels=labels, corner_colors=list(resolve_corners(corners)), fontsize=fontsize)
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, np.sqrt(3) / 2, n)
    gx, gy = np.meshgrid(xs, ys)
    # invert xy -> barycentric
    T = np.array([[_V[0, 0] - _V[2, 0], _V[1, 0] - _V[2, 0]], [_V[0, 1] - _V[2, 1], _V[1, 1] - _V[2, 1]]])
    Tinv = np.linalg.inv(T)
    d = np.stack([gx - _V[2, 0], gy - _V[2, 1]], -1)
    ab = d @ Tinv.T
    a, b = ab[..., 0], ab[..., 1]
    c = 1 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    comp = np.stack([a, b, c], -1)
    rgb = color(comp, corners=corners, gamma=gamma)
    img = np.dstack([rgb, inside.astype(float)])
    ax.imshow(img, extent=(0, 1, 0, np.sqrt(3) / 2), origin="lower", interpolation="bilinear", zorder=0)
    if points is not None:
        xy = barycentric_to_xy(points)
        pk = dict(s=3, color="#222222", alpha=0.5, linewidths=0)
        pk.update(point_kw or {})
        ax.scatter(xy[:, 0], xy[:, 1], zorder=3, **pk)
    return ax


def ternary_image(
    comp_img: ArrayLike,
    *,
    ax: Axes | None = None,
    corners: Corners = DEFAULT_CORNERS,
    labels: Sequence[str] = ("A", "B", "C"),
    inset: bool = True,
    title: str = "",
    gamma: float = 0.8,
) -> Axes:
    """Show a composition image of shape ``(H, W, 3)`` in ternary colours.

    Args:
        comp_img: Per-pixel compositions of shape ``(H, W, 3)``.
        ax: Axes to draw into; a new figure is created if ``None``.
        corners: Palette name or three corner colours.
        labels: Corner labels for the legend.
        inset: Add a ternary legend inset in the top-right corner.
        title: Axes title.
        gamma: As in :func:`color`.

    Returns:
        The image axes.
    """
    th = get_theme()
    if ax is None:
        from .figure import subplots

        _, ax = subplots(width="half", aspect=1.0)
    rgb = color(comp_img, corners=corners, gamma=gamma)
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=th.fs_title, pad=2.5)
    if inset:
        iax = ax.inset_axes([0.71, 0.71, 0.29, 0.29])
        iax.set_facecolor("white")
        iax.patch.set_alpha(0.85)
        ternary_legend(iax, corners=corners, labels=labels, gamma=gamma, fontsize=th.fs_annot)
    return ax


def ternary_map(
    lon: ArrayLike,
    lat: ArrayLike,
    comp: ArrayLike,
    *,
    ax: Axes | None = None,
    corners: Corners = DEFAULT_CORNERS,
    labels: Sequence[str] = ("A", "B", "C"),
    projection: str = "robinson",
    antarctica_cutoff: bool = True,
    extent: Sequence[float] | None = None,
    size: float = 6,
    inset: bool = True,
    land: bool = True,
    coastline: bool = True,
    title: str = "",
    gamma: float = 0.8,
    **kw: Any,
) -> tuple[Figure, Axes]:
    """Scatter points on a map, each coloured by its composition.

    Args:
        lon: Longitudes in degrees east, shape ``(N,)``.
        lat: Latitudes in degrees north, shape ``(N,)``.
        comp: Compositions of shape ``(N, 3)``.
        ax: Existing (cartopy) axes; a new figure is created if ``None``.
        corners: Palette name or three corner colours.
        labels: Corner labels for the legend.
        projection: Projection name, as in :func:`nicefigs.make_map_axes`.
        antarctica_cutoff: Crop the southern edge when ``extent`` is ``None``.
        extent: ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
        size: Marker area in points².
        inset: Add a ternary legend inset in the lower-left corner.
        land: Draw the land background.
        coastline: Draw coastlines.
        title: Axes title.
        gamma: As in :func:`color`.
        **kw: Forwarded to :meth:`matplotlib.axes.Axes.scatter`.

    Returns:
        ``(fig, ax)``.
    """
    from .maps import _apply_extent, _try_cartopy, draw_background, make_map_axes

    th = get_theme()
    if ax is None:
        from .figure import figure_size

        fig = plt.figure(figsize=figure_size("full", aspect=0.52), dpi=th.dpi)
        ax, ccrs = make_map_axes(fig, 111, projection=projection)
    else:
        fig = ax.figure
        ccrs = _try_cartopy() if hasattr(ax, "projection") else None
    draw_background(ax, ccrs, land=land, coastline=coastline)
    _apply_extent(ax, ccrs, extent, antarctica_cutoff and extent is None)
    cols = color(comp, corners=corners, gamma=gamma)
    skw = dict(s=size, linewidths=0, rasterized=True)
    if ccrs is not None:
        skw["transform"] = ccrs.PlateCarree()
    ax.scatter(np.asarray(lon), np.asarray(lat), c=cols, **skw, **kw)
    if title:
        from .panels import clean_title

        clean_title(ax, title)
    if inset:
        # Lower-left corner of a global map is open ocean, so the inset needs no backing.
        iax = ax.inset_axes([0.03, 0.06, 0.19, 0.19])
        iax.patch.set_alpha(0.0)
        ternary_legend(iax, corners=corners, labels=labels, gamma=gamma, fontsize=th.fs_annot - 0.5)
    return fig, ax
