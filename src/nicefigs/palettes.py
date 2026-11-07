"""Colormaps and palettes: named colormaps, land-cover classes, comparison roles and colour limits.

Colormaps resolve by name through :func:`cmap` (matplotlib names, the data-kind
names in :data:`DEFAULTS`, and :data:`CUSTOM_CMAPS`); land-cover products
through :func:`categorical`; comparison roles through :func:`role_style`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import ArrayLike

from .artists import ARTIST_PALETTES, _register
from .style import PALETTE_BRIGHT, PALETTE_HIGH_CONTRAST, PALETTE_MUTED, get_theme

#: Default colormap name by data kind; each key also resolves through :func:`cmap`.
DEFAULTS = {
    "sequential": "inferno",
    "sequential2": "Spectral_r",
    "intensity": "inferno",
    "diverging": "RdBu_r",
    "diverging2": "BrBG",
    "cyclic": "twilight",
    "rainbow": "Spectral_r",
}

# Lower-case aliases of matplotlib names.
_ALIASES = {
    "spectral_r": "Spectral_r",
    "spectral": "Spectral",
    "rdbu": "RdBu_r",
    "rdbu_r": "RdBu_r",
    "brbg": "BrBG",
    "brbg_r": "BrBG_r",
    "inferno": "inferno",
    "viridis": "viridis",
    "magma": "magma",
    "turbo": "turbo",
}


#: Colormaps beyond matplotlib's, resolvable by :func:`cmap`; extend in place.
CUSTOM_CMAPS: dict[str, mcolors.Colormap] = {
    # Diverging ramp, white at the centre: cool -> warm -> violet.
    "inside_outside": mcolors.LinearSegmentedColormap.from_list(
        "inside_outside",
        [
            "#003f5c",
            "#3d6d8e",
            "#8fb3c9",
            "#cfe0ea",
            "#ffffff",
            "#f3c1a0",
            "#dd8452",
            "#a34511",
            "#581c87",
        ],
    ),
}

#: Band edges (percent) of :func:`scorecard_diverging_cmap`: 12 edges bounding
#: 11 bands, quasi-logarithmic and symmetric about 0.
SCORECARD_DIVERGING_BOUNDARIES: list[float] = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]

# Band colours, most negative first.
_SCORECARD_DIVERGING_COLORS = [
    "#437AB5",
    "#669CC9",
    "#94BDD9",
    "#C0D5E9",
    "#DEE9F5",
    "#F2F2F2",
    "#F9DDCE",
    "#F1B398",
    "#EC8869",
    "#E25C45",
    "#C4372D",
]
_SCORECARD_DIVERGING_UNDER = "#26549A"
_SCORECARD_DIVERGING_OVER = "#9C2520"

CUSTOM_CMAPS["scorecard_diverging"] = mcolors.ListedColormap(
    _SCORECARD_DIVERGING_COLORS, name="scorecard_diverging"
)


def cmap(name: str | mcolors.Colormap, n: int | None = None) -> mcolors.Colormap:
    """Resolve a colormap by name.

    Args:
        name: A :data:`CUSTOM_CMAPS` or :data:`DEFAULTS` key, a matplotlib
            colormap name, or a ``Colormap`` (returned as is).
        n: Resample to ``n`` discrete levels.

    Raises:
        KeyError: Unknown name.
    """
    if isinstance(name, mcolors.Colormap):
        cm = name
    else:
        key = str(name)
        if key in CUSTOM_CMAPS:
            cm = CUSTOM_CMAPS[key]
        else:
            key = _ALIASES.get(key.lower(), key)
            if key in DEFAULTS:
                key = DEFAULTS[key]
                key = _ALIASES.get(key.lower(), key)
            cm = CUSTOM_CMAPS.get(key) or mpl.colormaps[key]
    if n is not None:
        cm = cm.resampled(n)
    return cm


# --------------------------------------------------------------------------- #
# Role palette
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Role:
    """How one role in a comparison figure (method under study, baseline, ...) is drawn.

    Attributes:
        color: Line, marker and marker-edge colour.
        marker: Matplotlib marker shape.
        filled: Whether markers are filled (``False`` draws them open).
        linestyle: Line style for lines.
        linewidth_scale: Line width as a multiple of the active theme's ``lw_line``.
        zorder: Drawing order; lower draws behind the data.
        band: Fill colour for a range drawn behind the data, or ``None``.
    """

    color: str
    marker: str
    filled: bool = True
    linestyle: str = "-"
    linewidth_scale: float = 1.0
    zorder: float = 2.0
    band: str | None = None


#: Style of each role. Every pair of role colours is at least 16 apart (OKLab ΔE × 100)
#: under normal, protan, deutan and tritan vision (Machado et al. 2009, severity 1).
#: ``null`` is below 3:1 contrast on white: label it directly, never by colour alone.
ROLES: dict[str, Role] = {
    # the method under study
    "primary": Role("#4477AA", "o"),
    # the competing method it is compared against
    "contrast": Role("#D95F02", "s"),
    # established baselines, told apart by label rather than by hue
    "baseline": Role("#3A3A3A", "D"),
    # chance level or control: open markers, or a band behind the data
    "null": Role("#AAAAAA", "o", filled=False, zorder=1.0, band="#E6E6E6"),
    # truth, ceilings and targets: thin dashed lines, stars for points
    "reference": Role("#000000", "*", linestyle="--", linewidth_scale=0.5, zorder=1.5),
}

_ROLE_KINDS = ("marker", "line", "both", "band")


def _role(role: str) -> Role:
    try:
        return ROLES[role]
    except KeyError as exc:
        raise KeyError(f"Unknown role {role!r}. Available: {sorted(ROLES)}") from exc


def role_color(role: str, *, band: bool = False) -> str:
    """Hex colour of ``role`` in :data:`ROLES`, or of its band fill if ``band``.

    Raises:
        KeyError: ``role`` is not in :data:`ROLES`.
        ValueError: ``band`` is requested for a role without a band colour.
    """
    r = _role(role)
    if not band:
        return r.color
    if r.band is None:
        raise ValueError(f"Role {role!r} has no band colour.")
    return r.band


def role_style(
    role: str, kind: str = "marker", *, filled: bool | None = None, **overrides: Any
) -> dict[str, Any]:
    """Keyword arguments that draw ``role`` with ``ax.plot``, ``ax.errorbar`` or ``ax.fill_between``.

    Line widths scale with the active theme's ``lw_line``; marker size is left
    to the caller.

    Args:
        role: A key of :data:`ROLES`.
        kind: ``"marker"`` (markers, no line), ``"line"`` (line, no markers),
            ``"both"``, or ``"band"`` (fill keywords for ``fill_between`` or
            ``axhspan``, drawn behind the role's markers).
        filled: Override the role's marker fill. ``False`` gives the open
            variant, e.g. for a sensitivity run of the same method.
        **overrides: Matplotlib keywords that replace the role's defaults.

    Returns:
        A new dict of matplotlib keyword arguments.

    Raises:
        KeyError: ``role`` is not in :data:`ROLES`.
        ValueError: Unknown ``kind``, or ``kind="band"`` for a role without a band.

    Example::

        ax.plot(x, y, **nf.role_style("primary", "both"))
        ax.fill_between(x, lo, hi, **nf.role_style("null", "band"))
    """
    if kind not in _ROLE_KINDS:
        raise ValueError(f"Unknown kind {kind!r}. Expected one of {_ROLE_KINDS}.")
    r = _role(role)
    if kind == "band":
        band = {"color": role_color(role, band=True), "linewidth": 0, "zorder": r.zorder - 0.5}
        return {**band, **overrides}
    kw: dict[str, Any] = {"color": r.color, "zorder": r.zorder}
    if kind == "marker":
        kw["linestyle"] = "none"
    else:
        kw["linestyle"] = r.linestyle
        kw["linewidth"] = r.linewidth_scale * get_theme().lw_line
    if kind == "line":
        kw["marker"] = "none"
    else:
        fill = r.filled if filled is None else filled
        kw.update(marker=r.marker, markeredgecolor=r.color, markerfacecolor=r.color if fill else "none")
    return {**kw, **overrides}


def diverging_norm(vmin: float, vmax: float, center: float = 0.0) -> mcolors.TwoSlopeNorm:
    """A ``TwoSlopeNorm`` mapping ``center`` to the colormap's midpoint, for anomaly maps.

    ``vmin`` and ``vmax`` are widened if needed so that ``vmin < center < vmax``.
    """
    vmin = min(vmin, center - 1e-9)
    vmax = max(vmax, center + 1e-9)
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def symmetric_limits(values: ArrayLike, *, percentile: float = 98.0) -> tuple[float, float]:
    """Colour limits ``(-m, m)`` with ``m`` the ``percentile`` of ``|values|``.

    Non-finite values are ignored; no finite values give ``(-1, 1)``.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    m = float(np.percentile(np.abs(a), percentile))
    m = max(m, 1e-12)
    return -m, m


def robust_limits(values: ArrayLike, *, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """Colour limits at the ``low`` and ``high`` percentiles of ``values``.

    Non-finite values are ignored; no finite values give ``(0, 1)``.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(a, [low, high])
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Geoscience-specific continuous colormaps
# --------------------------------------------------------------------------- #


def elevation_cmap(
    *,
    sea_level: float = 0.0,
    vmin: float = -8000.0,
    vmax: float = 6000.0,
    name: str = "nicefigs_elevation",
) -> tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
    """Hypsometric elevation colormap and its norm.

    Blues from deep to shallow below ``sea_level``; green, tan, brown and white
    above.

    Args:
        sea_level: Elevation of the land/sea colour break (m).
        vmin: Lowest elevation (m).
        vmax: Highest elevation (m).
        name: Name of the colormap.

    Returns:
        ``(cmap, norm)``; use both together so the break lands at ``sea_level``.
    """
    ocean = [
        (0.00, "#08306b"),
        (0.60, "#4292c6"),
        (1.00, "#c6dbef"),
    ]
    land = [
        (0.00, "#1a7a3c"),
        (0.18, "#8cc665"),
        (0.40, "#e8d98a"),
        (0.62, "#c9a24b"),
        (0.80, "#8f6b3f"),
        (0.92, "#c9c0b8"),
        (1.00, "#ffffff"),
    ]
    span = vmax - vmin
    break_frac = (sea_level - vmin) / span
    stops = []
    for pos, col in ocean:
        stops.append((pos * break_frac, col))
    for pos, col in land:
        stops.append((break_frac + pos * (1 - break_frac), col))
    # Ocean and land stops meet at break_frac; keep one stop per position.
    seen = {}
    for pos, col in stops:
        seen[round(pos, 6)] = col
    stops = sorted(seen.items())
    cm = mcolors.LinearSegmentedColormap.from_list(name, stops)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cm, norm


# --------------------------------------------------------------------------- #
# Scorecard diverging colormap
# --------------------------------------------------------------------------- #


def scorecard_diverging_cmap(
    boundaries: list[float] | None = None,
) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Discrete, quasi-logarithmic diverging ``(cmap, norm)`` for skill scorecards.

    This is **not** "the WeatherBench colormap": WeatherBench2's released
    ``weatherbench2/visualization.py`` colours scorecards with continuous
    ``RdBu_r`` at ``vmin=-100, vmax=100`` — no discrete bands, no
    ``BoundaryNorm``, no extend triangles. The bands and colours here were
    instead sampled pixel-wise from WeatherBench2's *published website figure*,
    which underwent a separate design pass never released as code. The sampled
    colours are confirmed not to be ``RdBu_r`` under any sampling: the worst
    per-channel error is 61/255 at equal band centres and 93/255 at 11 equal
    steps. So this is a discrete quasi-logarithmic diverging scale,
    WeatherBench2-*inspired* — not a reproduction of WeatherBench2's own code.

    The quasi-log spacing (``1, 2, 5, 10, 20, 50`` per side) resolves the
    +/-1-5% range where real forecast-skill differences cluster, while still
    reaching +/-50%; a continuous linear +/-100% scale (as WB2's code uses)
    flattens that whole cluster to near-uniform pale colour. The centre band is
    light grey (``#F2F2F2``), not white, and the under/over extend colours
    (``#26549A`` / ``#9C2520``) are deliberately distinct from the outermost
    band colours — the extend triangles read as their own colour, not a repeat
    of the band next to them.

    Args:
        boundaries: 12 band edges (11 bands). Defaults to
            :data:`SCORECARD_DIVERGING_BOUNDARIES`.

    Returns:
        ``(cmap, norm)``: a ``ListedColormap`` with ``set_under``/``set_over``
        applied, and a matching ``BoundaryNorm`` — ready for
        ``imshow``/``pcolormesh``/``colorbar`` with ``extend="both"``.
    """
    boundaries = list(boundaries) if boundaries is not None else list(SCORECARD_DIVERGING_BOUNDARIES)
    if len(boundaries) != len(_SCORECARD_DIVERGING_COLORS) + 1:
        raise ValueError(
            f"boundaries must have {len(_SCORECARD_DIVERGING_COLORS) + 1} edges "
            f"for {len(_SCORECARD_DIVERGING_COLORS)} bands, got {len(boundaries)}"
        )
    base = mcolors.ListedColormap(_SCORECARD_DIVERGING_COLORS, name="scorecard_diverging")
    cm = base.with_extremes(under=_SCORECARD_DIVERGING_UNDER, over=_SCORECARD_DIVERGING_OVER)
    cm.name = "scorecard_diverging"
    norm = mcolors.BoundaryNorm(boundaries, cm.N)
    return cm, norm


# --------------------------------------------------------------------------- #
# Categorical palettes (land-cover products)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Categorical:
    """A categorical palette mapping integer class codes to names and colours.

    Attributes:
        key: Palette name.
        names: Display name per class code.
        colors: Hex colour per class code.
    """

    key: str
    names: dict[int, str]
    colors: dict[int, str]

    @property
    def codes(self) -> list[int]:
        """Class codes in ascending order."""
        return sorted(self.colors)

    def cmap_norm(self) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        """``(cmap, norm)`` mapping each class code to its colour, for ``ax.imshow(codes, cmap=cmap, norm=norm)``."""
        codes = self.codes
        cols = [self.colors[c] for c in codes]
        cm = mcolors.ListedColormap(cols)
        bounds = [codes[0] - 0.5] + [0.5 * (a + b) for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 0.5]
        norm = mcolors.BoundaryNorm(bounds, cm.N)
        return cm, norm

    def to_rgb(self, arr: ArrayLike, *, default: str = "#dddddd") -> np.ndarray:
        """Convert class codes of shape ``(H, W)`` to a float32 RGB image ``(H, W, 3)``.

        Args:
            arr: Class codes.
            default: Colour of codes not in the palette.
        """
        arr = np.asarray(arr)
        out = np.tile(np.asarray(mcolors.to_rgb(default), dtype=np.float32), (*arr.shape, 1))
        for code, hexcol in self.colors.items():
            m = arr == code
            if m.any():
                out[m] = mcolors.to_rgb(hexcol)
        return out

    def legend_handles(self, codes: list[int] | None = None) -> list[Patch]:
        """Legend patches, one per code in ``codes`` (default: all) that the palette defines."""
        codes = codes if codes is not None else self.codes
        return [
            Patch(facecolor=self.colors[c], edgecolor="none", label=self.names.get(c, str(c)))
            for c in codes
            if c in self.colors
        ]


_ESA_WC_COLORS = {
    10: "#006400",
    20: "#ffbb22",
    30: "#ffff4c",
    40: "#f096ff",
    50: "#fa0000",
    60: "#b4b4b4",
    70: "#f0f0f0",
    80: "#0064c8",
    90: "#0096a0",
    95: "#00cf75",
    100: "#fae6a0",
}
_ESA_WC_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse veg.",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

_ESRI_COLORS = {
    1: "#1A5BAB",
    2: "#358221",
    3: "#A7D282",
    4: "#87D19E",
    5: "#E49635",
    6: "#DFC35A",
    7: "#ED022A",
    8: "#EDE9E4",
    9: "#F2FAFF",
}
_ESRI_NAMES = {
    1: "Water",
    2: "Trees",
    3: "Grass",
    4: "Flooded veg.",
    5: "Crops",
    6: "Scrub/Shrub",
    7: "Built area",
    8: "Bare ground",
    9: "Snow/Ice",
}

_MODIS_COLORS = {
    1: "#05450a",
    2: "#086a10",
    3: "#54a708",
    4: "#78d203",
    5: "#009900",
    6: "#c6b044",
    7: "#dcd159",
    8: "#dade48",
    9: "#fbff13",
    10: "#b6ff05",
    11: "#27ff87",
    12: "#c24f44",
    13: "#a5a5a5",
    14: "#ff6d4c",
    15: "#f9ffa4",
    16: "#69fff8",
    17: "#1c0dff",
}
_MODIS_NAMES = {
    1: "Evergreen Needleleaf",
    2: "Evergreen Broadleaf",
    3: "Deciduous Needleleaf",
    4: "Deciduous Broadleaf",
    5: "Mixed Forest",
    6: "Closed Shrubland",
    7: "Open Shrubland",
    8: "Woody Savanna",
    9: "Savanna",
    10: "Grassland",
    11: "Permanent Wetland",
    12: "Cropland",
    13: "Urban / Built-up",
    14: "Cropland/Nat. Mosaic",
    15: "Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
}

CATEGORICAL: dict[str, Categorical] = {
    "esa_worldcover": Categorical("esa_worldcover", _ESA_WC_NAMES, _ESA_WC_COLORS),
    "esri_lulc": Categorical("esri_lulc", _ESRI_NAMES, _ESRI_COLORS),
    "modis_igbp": Categorical("modis_igbp", _MODIS_NAMES, _MODIS_COLORS),
}


def categorical(name: str) -> Categorical:
    """Look up a categorical palette in :data:`CATEGORICAL` by name, e.g. ``"esa_worldcover"``.

    Raises:
        KeyError: Unknown name.
    """
    try:
        return CATEGORICAL[name]
    except KeyError as exc:
        raise KeyError(f"Unknown categorical palette {name!r}. Known: {sorted(CATEGORICAL)}") from exc


def qualitative(n: int, palette: str = "nice") -> list[str]:
    """``n`` qualitative colours from a named palette.

    Args:
        n: Number of colours; the palette repeats if ``n`` exceeds its length.
        palette: ``"bright"``, ``"muted"`` or ``"high_contrast"`` (colour-blind
            safe), or a key of :data:`nicefigs.ARTIST_PALETTES` such as
            ``"nice"`` or ``"hockney"``.

    Returns:
        Hex colour strings.

    Raises:
        KeyError: ``palette`` is not a known palette name.
    """
    pals = {"bright": PALETTE_BRIGHT, "muted": PALETTE_MUTED, "high_contrast": PALETTE_HIGH_CONTRAST}
    pals.update(ARTIST_PALETTES)
    try:
        base = pals[palette]
    except KeyError as exc:
        raise KeyError(f"Unknown palette {palette!r}. Available: {sorted(pals)}") from exc
    if n <= len(base):
        return list(base[:n])
    return [base[i % len(base)] for i in range(n)]


_register(CUSTOM_CMAPS)
