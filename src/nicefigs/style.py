"""Themes: frozen bundles of font sizes (pt), line widths (pt) and colours.

Because font sizes and figure sizes are both physical units, a figure rendered
at a requested physical size (see :mod:`nicefigs.figure`) keeps its fonts at
exactly the theme's sizes.

Example::

    import nicefigs as nf

    nf.set_theme("paper")            # globally

    with nf.style("talk"):           # or temporarily
        fig, ax = plt.subplots()
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

from .artists import NICE

# --------------------------------------------------------------------------- #
# Colour palettes
# --------------------------------------------------------------------------- #

#: Paul Tol "bright" — colour-blind-safe qualitative palette (7 colours).
PALETTE_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

#: Paul Tol "muted" — colour-blind-safe, softer (9 colours + grey).
PALETTE_MUTED = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]

#: Paul Tol "high-contrast" (3 colours + black/white), good for line plots.
PALETTE_HIGH_CONTRAST = ["#004488", "#DDAA33", "#BB5566", "#000000"]

#: Semantic colours reused across the house style.
COLORS = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "gray": "#BBBBBB",
    # semantic aliases
    "primary": "#2166ac",
    "accent": "#d95f02",
    "good": "#1b9e77",
    "bad": "#d7191c",
    "neutral": "#666666",
    "muted": "#bdbdbd",
    "highlight": "#d95f02",
    "land": "#f0f0f0",
    "ocean": "#eaf1f5",
    "coast": "#888888",
}


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Theme:
    """A frozen bundle of styling parameters.

    Font sizes (``fs_*``), line widths (``lw_*``) and ``tick_len`` are in
    points. Use :meth:`from_base` to derive every font size from one base
    size, or :meth:`scaled` to scale an existing theme.

    Attributes:
        name: Preset name.
        base: Base (body) font size the ``fs_*`` sizes derive from.
        palette: Categorical colour cycle (hex strings).
        grid_alpha: Opacity of grid lines when ``grid`` is on.
        text_color: Colour of all text.
        spine_color: Colour of spines and ticks.
        font_family: Matplotlib ``font.family``.
        sans_serif: Font fallback list for ``font.sans-serif``.
        spines_topright: Draw the top and right spines.
        grid: Draw an axes grid by default.
        dpi: Screen resolution (dots per inch).
        savedpi: Resolution for raster output of :func:`nicefigs.save`.
    """

    name: str = "paper"
    base: float = 7.0

    fs_body: float = 6.5
    fs_panel_label: float = 7.8
    fs_title: float = 7.0
    fs_panel_title: float = 7.2
    fs_axis_label: float = 6.5
    fs_tick: float = 5.5
    fs_legend: float = 6.0
    fs_annot: float = 5.5

    lw_axes: float = 0.5
    lw_grid: float = 0.5
    lw_line: float = 1.2
    lw_tick: float = 0.4
    tick_len: float = 2.0

    palette: list[str] = field(default_factory=lambda: list(NICE))
    grid_alpha: float = 0.15
    text_color: str = "#222222"
    spine_color: str = "#222222"

    font_family: str = "sans-serif"
    sans_serif: tuple[str, ...] = ("Helvetica", "Arial", "DejaVu Sans", "sans-serif")

    spines_topright: bool = False
    grid: bool = False
    dpi: int = 150
    savedpi: int = 300

    @classmethod
    def from_base(cls, base: float, **overrides: Any) -> Theme:
        """Build a theme whose font sizes scale from one ``base`` size.

        Args:
            base: Body font size (pt).
            **overrides: Field values that replace the derived ones.

        Returns:
            The new theme.
        """
        sizes = dict(
            base=base,
            fs_body=base,
            fs_panel_label=round(base * 1.2, 2),
            fs_title=round(base * 1.08, 2),
            fs_panel_title=round(base * 1.1, 2),
            fs_axis_label=round(base * 1.00, 2),
            fs_tick=round(base * 0.85, 2),
            fs_legend=round(base * 0.92, 2),
            fs_annot=round(base * 0.85, 2),
        )
        sizes.update(overrides)
        return cls(**sizes)

    def scaled(self, factor: float) -> Theme:
        """Return a copy with every font size multiplied by ``factor``."""
        return replace(
            self,
            base=self.base * factor,
            fs_body=self.fs_body * factor,
            fs_panel_label=self.fs_panel_label * factor,
            fs_title=self.fs_title * factor,
            fs_panel_title=self.fs_panel_title * factor,
            fs_axis_label=self.fs_axis_label * factor,
            fs_tick=self.fs_tick * factor,
            fs_legend=self.fs_legend * factor,
            fs_annot=self.fs_annot * factor,
        )

    def with_(self, **overrides: Any) -> Theme:
        """Return a copy with the given fields overridden.

        Args:
            **overrides: Field names and their new values.
        """
        return replace(self, **overrides)

    def rc(self) -> dict[str, Any]:
        """Render this theme as a matplotlib ``rcParams`` dict."""
        return {
            "figure.dpi": self.dpi,
            "savefig.dpi": self.savedpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": self.font_family,
            "font.sans-serif": list(self.sans_serif),
            "font.size": self.fs_body,
            "text.color": self.text_color,
            "axes.edgecolor": self.spine_color,
            "axes.labelcolor": self.text_color,
            "xtick.color": self.spine_color,
            "ytick.color": self.spine_color,
            "xtick.labelcolor": self.text_color,
            "ytick.labelcolor": self.text_color,
            "axes.titlesize": self.fs_title,
            "axes.labelsize": self.fs_axis_label,
            "xtick.labelsize": self.fs_tick,
            "ytick.labelsize": self.fs_tick,
            "legend.fontsize": self.fs_legend,
            "figure.titlesize": self.fs_title,
            "axes.linewidth": self.lw_axes,
            "lines.linewidth": self.lw_line,
            "grid.linewidth": self.lw_grid,
            "xtick.major.width": self.lw_tick,
            "ytick.major.width": self.lw_tick,
            "xtick.minor.width": self.lw_tick * 0.7,
            "ytick.minor.width": self.lw_tick * 0.7,
            "xtick.major.size": self.tick_len,
            "ytick.major.size": self.tick_len,
            "xtick.minor.size": self.tick_len * 0.6,
            "ytick.minor.size": self.tick_len * 0.6,
            "xtick.major.pad": 1.8,
            "ytick.major.pad": 1.8,
            "axes.labelpad": 2.0,
            "axes.titlepad": 3.0,
            "axes.spines.top": self.spines_topright,
            "axes.spines.right": self.spines_topright,
            "axes.grid": self.grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.borderaxespad": 0.4,
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": mpl.cycler(color=self.palette),
            "image.cmap": "inferno",
        }


# --------------------------------------------------------------------------- #
# Named theme presets
# --------------------------------------------------------------------------- #

THEMES: dict[str, Theme] = {
    # Dense multi-panel figures; the field defaults.
    "paper": Theme(name="paper"),
    # Single-panel / half-column figures where a touch more air reads better.
    "paper-large": Theme.from_base(8.5, name="paper-large"),
    # Talks & slides.
    "talk": Theme.from_base(13.0, name="talk", lw_axes=0.9, lw_line=2.0, tick_len=3.5),
    # Posters.
    "poster": Theme.from_base(20.0, name="poster", lw_axes=1.2, lw_line=2.8, tick_len=4.5),
}

_active_theme: Theme = THEMES["paper"]


def get_theme(theme: str | Theme | None = None) -> Theme:
    """Resolve a theme name, a :class:`Theme`, or ``None`` (the active theme).

    Raises:
        KeyError: ``theme`` is not a key of :data:`THEMES`.
    """
    if theme is None:
        return _active_theme
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError as exc:
        raise KeyError(f"Unknown theme {theme!r}. Available: {sorted(THEMES)}") from exc


def set_theme(theme: str | Theme = "paper") -> Theme:
    """Set the active theme globally and apply its rcParams.

    Args:
        theme: Preset name (a key of :data:`THEMES`) or a :class:`Theme`.

    Returns:
        The resolved theme.
    """
    global _active_theme
    _active_theme = get_theme(theme)
    plt.rcParams.update(_active_theme.rc())
    return _active_theme


@contextlib.contextmanager
def style(theme: str | Theme | None = None, **rc_overrides: Any) -> Iterator[Theme]:
    """Apply a theme within a ``with`` block, restoring the previous one on exit.

    The theme is also the active one (:func:`get_theme`) inside the block.

    Args:
        theme: Preset name, a :class:`Theme`, or ``None`` for the active theme.
        **rc_overrides: Extra ``rcParams``, e.g.
            ``nf.style("paper", **{"axes.grid": True})``.

    Yields:
        The resolved theme.
    """
    global _active_theme
    resolved = get_theme(theme)
    prev = _active_theme
    rc = resolved.rc()
    rc.update(rc_overrides)
    _active_theme = resolved
    try:
        with plt.rc_context(rc):
            yield resolved
    finally:
        _active_theme = prev
