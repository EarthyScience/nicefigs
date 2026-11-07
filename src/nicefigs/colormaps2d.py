"""Two-dimensional colormaps that encode two continuous variables in one colour.

Three encoders are provided, each with a ``*_legend`` renderer for its square key:

- :func:`bivariate_cmap` — bilinear blend of four corner colours.
- :func:`hsv_cmap` — hue along one axis, saturation along the other, at fixed value.
- :func:`flat_cmap` — a constant-lightness CIE-Lab plane (perceptually even).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import ArrayLike

from .style import get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# --------------------------------------------------------------------------- #
# Bivariate (4-corner bilinear)
# --------------------------------------------------------------------------- #

#: Default 2D corners: (x-low,y-low), (x-high,y-low), (x-low,y-high), (x-high,y-high).
BIVARIATE_CORNERS: dict[str, str] = {"ll": "#e8e8e8", "hl": "#e8b24a", "lh": "#5ac8c8", "hh": "#3b1f66"}


def normalize01(
    values: ArrayLike, *, method: Literal["quantile", "linear"] = "quantile", clip: bool = True
) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale values into ``[0, 1]`` for use with a 2D colormap.

    Args:
        values: Values of any shape; NaNs are ignored when choosing the range.
        method: ``"quantile"`` maps the 5th/95th percentiles to 0/1, ``"linear"``
            the minimum/maximum.
        clip: Clip the scaled values to ``[0, 1]``.

    Returns:
        ``(scaled, (lo, hi))``: the scaled array (same shape as ``values``) and the
        data range mapped to 0 and 1, for legend tick labels.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr), (0.0, 1.0)
    if method == "quantile":
        lo, hi = float(np.nanpercentile(arr, 5)), float(np.nanpercentile(arr, 95))
    else:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (arr - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) if clip else scaled), (lo, hi)


def bivariate_cmap(
    x: ArrayLike, y: ArrayLike, *, corners: Mapping[str, str] | None = None, clip: bool = True
) -> np.ndarray:
    """Colour ``(x, y)`` pairs by bilinear blending of four corner colours.

    Args:
        x: First variable, scaled to ``[0, 1]``.
        y: Second variable, scaled to ``[0, 1]``; same shape as ``x``.
        corners: Colours keyed ``"ll"``, ``"hl"``, ``"lh"``, ``"hh"`` (x-low/high,
            y-low/high). Defaults to :data:`BIVARIATE_CORNERS`.
        clip: Clip ``x`` and ``y`` to ``[0, 1]`` first.

    Returns:
        RGB array of shape ``x.shape + (3,)`` in ``[0, 1]``.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if clip:
        xs, ys = np.clip(xs, 0, 1), np.clip(ys, 0, 1)
    c = {k: np.asarray(mcolors.to_rgb(v)) for k, v in (corners or BIVARIATE_CORNERS).items()}
    w_ll = (1 - xs) * (1 - ys)
    w_hl = xs * (1 - ys)
    w_lh = (1 - xs) * ys
    w_hh = xs * ys
    rgb = (
        w_ll[..., None] * c["ll"]
        + w_hl[..., None] * c["hl"]
        + w_lh[..., None] * c["lh"]
        + w_hh[..., None] * c["hh"]
    )
    return np.clip(rgb, 0, 1)


def hsv_cmap(
    hue: ArrayLike,
    sat: ArrayLike,
    *,
    value: float = 0.88,
    sat_range: tuple[float, float] = (0.5, 1.0),
    hue_range: tuple[float, float] = (0.55, 1.0),
    clip: bool = True,
) -> np.ndarray:
    """Colour pairs by hue along one variable and saturation along the other.

    Args:
        hue: Variable mapped to hue, scaled to ``[0, 1]``.
        sat: Variable mapped to saturation, scaled to ``[0, 1]``; same shape as ``hue``.
        value: Fixed HSV value (brightness) in ``[0, 1]``.
        sat_range: Saturation at ``sat`` = 0 and 1.
        hue_range: Hue (fraction of the colour wheel) at ``hue`` = 0 and 1.
        clip: Clip ``hue`` and ``sat`` to ``[0, 1]`` first.

    Returns:
        RGB array of shape ``hue.shape + (3,)`` in ``[0, 1]``.
    """
    hs = np.asarray(hue, dtype=float)
    ss = np.asarray(sat, dtype=float)
    if clip:
        hs, ss = np.clip(hs, 0, 1), np.clip(ss, 0, 1)
    h = hue_range[0] + hs * (hue_range[1] - hue_range[0])
    s = sat_range[0] + ss * (sat_range[1] - sat_range[0])
    v = np.full_like(h, value, dtype=float)
    return mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1))


# --------------------------------------------------------------------------- #
# Constant-lightness "flat" CIE-Lab wheel
# --------------------------------------------------------------------------- #

_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_XYZ = np.array([0.95047, 1.0, 1.08883])
_LAB_DELTA = 6.0 / 29.0


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t):
        return np.where(t > _LAB_DELTA, t**3, 3 * _LAB_DELTA**2 * (t - 4.0 / 29.0))

    xyz = np.stack([_WHITE_XYZ[0] * finv(fx), _WHITE_XYZ[1] * finv(fy), _WHITE_XYZ[2] * finv(fz)], -1)
    linear = xyz @ _XYZ_TO_RGB.T
    gamma = np.where(
        linear > 0.0031308, 1.055 * np.power(np.clip(linear, 0, None), 1 / 2.4) - 0.055, 12.92 * linear
    )
    return np.clip(gamma, 0, 1)


def flat_cmap(
    x: ArrayLike,
    y: ArrayLike,
    *,
    L: float = 95.0,
    chroma: float = 33.0,
    rotation_deg: float = 0.0,
    transpose: bool = False,
) -> np.ndarray:
    """Colour pairs on a constant-lightness CIE-Lab plane.

    ``x`` maps to a* (green to red) and ``y`` to b* (blue to yellow), so every
    colour has the same perceived lightness.

    Args:
        x: First variable, scaled to ``[0, 1]`` (clipped).
        y: Second variable, scaled to ``[0, 1]`` (clipped); same shape as ``x``.
        L: CIE-Lab lightness (0-100).
        chroma: Maximum a*/b* magnitude; larger is more saturated.
        rotation_deg: Rotation of the a*/b* plane in degrees.
        transpose: Swap the roles of ``x`` and ``y``.

    Returns:
        ``float32`` RGB array of shape ``x.shape + (3,)`` in ``[0, 1]``.
    """
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    y = np.clip(np.asarray(y, dtype=float), 0, 1)
    if transpose:
        x, y = y, x
    a = chroma * (2 * x - 1)
    b = chroma * (2 * y - 1)
    if rotation_deg:
        th = np.deg2rad(rotation_deg)
        ca, sa = np.cos(th), np.sin(th)
        a, b = a * ca - b * sa, a * sa + b * ca
    lab = np.stack([np.full_like(a, L), a, b], axis=-1)
    return _lab_to_srgb(lab).astype(np.float32)


# --------------------------------------------------------------------------- #
# Legends
# --------------------------------------------------------------------------- #


def _legend_square(
    ax: Axes,
    rgb: np.ndarray,
    x_label: str,
    y_label: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    fontsize: float,
) -> None:
    ax.imshow(rgb, origin="lower", extent=(0, 1, 0, 1), aspect="equal", interpolation="bilinear")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"{x_range[0]:.2g}", f"{x_range[1]:.2g}"], fontsize=fontsize)
    ax.set_yticklabels([f"{y_range[0]:.2g}", f"{y_range[1]:.2g}"], fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(length=1.5, width=0.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)


def bivariate_legend(
    ax: Axes,
    *,
    x_label: str = "",
    y_label: str = "",
    x_range: tuple[float, float] = (0, 1),
    y_range: tuple[float, float] = (0, 1),
    corners: Mapping[str, str] | None = None,
    n: int = 16,
    fontsize: float | None = None,
) -> None:
    """Draw the :func:`bivariate_cmap` key as a square image on ``ax``.

    Args:
        ax: Axes to draw into, typically a small inset.
        x_label: Label of the horizontal (``x``) axis.
        y_label: Label of the vertical (``y``) axis.
        x_range: Data values at the left/right edges, shown as tick labels.
        y_range: Data values at the bottom/top edges, shown as tick labels.
        corners: Corner colours, as in :func:`bivariate_cmap`.
        n: Grid resolution of the key per side.
        fontsize: Label size in points; defaults to the theme's annotation size.
    """
    th = get_theme()
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    rgb = bivariate_cmap(gx.ravel(), gy.ravel(), corners=corners).reshape(n, n, 3)
    _legend_square(ax, rgb, x_label, y_label, x_range, y_range, fontsize or th.fs_annot)


def flat_legend(
    ax: Axes,
    *,
    x_label: str = "",
    y_label: str = "",
    x_range: tuple[float, float] = (0, 1),
    y_range: tuple[float, float] = (0, 1),
    L: float = 95.0,
    chroma: float = 33.0,
    rotation_deg: float = 0.0,
    transpose: bool = False,
    n: int = 128,
    fontsize: float | None = None,
) -> None:
    """Draw the :func:`flat_cmap` key as a square image on ``ax``.

    Args:
        ax: Axes to draw into, typically a small inset.
        x_label: Label of the horizontal (``x``) axis.
        y_label: Label of the vertical (``y``) axis.
        x_range: Data values at the left/right edges, shown as tick labels.
        y_range: Data values at the bottom/top edges, shown as tick labels.
        L: CIE-Lab lightness, as in :func:`flat_cmap`.
        chroma: Maximum a*/b* magnitude, as in :func:`flat_cmap`.
        rotation_deg: Rotation of the a*/b* plane, as in :func:`flat_cmap`.
        transpose: Swap the two axes, as in :func:`flat_cmap`.
        n: Grid resolution of the key per side.
        fontsize: Label size in points; defaults to the theme's annotation size.
    """
    th = get_theme()
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    rgb = flat_cmap(
        gx.ravel(), gy.ravel(), L=L, chroma=chroma, rotation_deg=rotation_deg, transpose=transpose
    ).reshape(n, n, 3)
    _legend_square(ax, rgb, x_label, y_label, x_range, y_range, fontsize or th.fs_annot)


def hsv_legend(
    ax: Axes,
    *,
    x_label: str = "",
    y_label: str = "",
    x_range: tuple[float, float] = (0, 1),
    y_range: tuple[float, float] = (0, 1),
    value: float = 0.88,
    sat_range: tuple[float, float] = (0.5, 1.0),
    hue_range: tuple[float, float] = (0.55, 1.0),
    n: int = 64,
    fontsize: float | None = None,
) -> None:
    """Draw the :func:`hsv_cmap` key as a square image on ``ax``.

    Args:
        ax: Axes to draw into, typically a small inset.
        x_label: Label of the horizontal (hue) axis.
        y_label: Label of the vertical (saturation) axis.
        x_range: Data values at the left/right edges, shown as tick labels.
        y_range: Data values at the bottom/top edges, shown as tick labels.
        value: HSV value, as in :func:`hsv_cmap`.
        sat_range: Saturation range, as in :func:`hsv_cmap`.
        hue_range: Hue range, as in :func:`hsv_cmap`.
        n: Grid resolution of the key per side.
        fontsize: Label size in points; defaults to the theme's annotation size.
    """
    th = get_theme()
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    rgb = hsv_cmap(gx.ravel(), gy.ravel(), value=value, sat_range=sat_range, hue_range=hue_range).reshape(
        n, n, 3
    )
    _legend_square(ax, rgb, x_label, y_label, x_range, y_range, fontsize or th.fs_annot)


__all__ = [
    "BIVARIATE_CORNERS",
    "normalize01",
    "bivariate_cmap",
    "hsv_cmap",
    "flat_cmap",
    "bivariate_legend",
    "flat_legend",
    "hsv_legend",
]
