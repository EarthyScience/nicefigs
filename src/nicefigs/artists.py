"""Artist-inspired categorical palettes and colormaps.

Hex values of the MetBrewer and MoMAColors palettes are taken verbatim from
`MetBrewer <https://github.com/BlakeRMills/MetBrewer>`_ and
`MoMAColors <https://github.com/BlakeRMills/MoMAColors>`_; ``"hockney"`` is
sampled from David Hockney's *A Bigger Grand Canyon* (1998), and ``"nice"``
(the default colour cycle) leads with a bright teal followed by the Hockney
colours. Continuous versions are registered for :func:`nicefigs.cmap`, e.g.
``cmap("hockney_canyon")`` or ``cmap("hiroshige")``.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors

if TYPE_CHECKING:
    from matplotlib.typing import ColorType

#: MetBrewer palettes by name.
METBREWER: dict[str, list[str]] = {
    "hokusai1": ["#6d2f20", "#b75347", "#df7e66", "#e09351", "#edc775", "#94b594", "#224b5e"],
    "hokusai2": ["#abc9c8", "#72aeb6", "#4692b0", "#2f70a1", "#134b73", "#0a3351"],
    "hokusai3": ["#d8d97a", "#95c36e", "#74c8c3", "#5a97c1", "#295384", "#0a2e57"],
    "hiroshige": [
        "#e76254",
        "#ef8a47",
        "#f7aa58",
        "#ffd06f",
        "#ffe6b7",
        "#aadce0",
        "#72bcd5",
        "#528fad",
        "#376795",
        "#1e466e",
    ],
    "klimt": ["#df9ed4", "#c93f55", "#eacc62", "#469d76", "#3c4b99", "#924099"],
    "vangogh1": ["#2c2d54", "#434475", "#6b6ca3", "#969bc7", "#87bcbd", "#89ab7c", "#6f9954"],
    "vangogh2": ["#bd3106", "#d9700e", "#e9a00e", "#eebe04", "#5b7314", "#c3d6ce", "#89a6bb", "#454b87"],
    "vangogh3": ["#e7e5cc", "#c2d6a4", "#9cc184", "#669d62", "#3c7c3d", "#1f5b25", "#1e3d14", "#192813"],
    "monet": [
        "#4e6d58",
        "#749e89",
        "#abccbe",
        "#e3cacf",
        "#c399a2",
        "#9f6e71",
        "#41507b",
        "#7d87b2",
        "#c2cae3",
    ],
    "renoir": [
        "#17154f",
        "#2f357c",
        "#6c5d9e",
        "#9d9cd5",
        "#b0799a",
        "#f6b3b0",
        "#e48171",
        "#bf3729",
        "#e69b00",
        "#f5bb50",
        "#ada43b",
        "#355828",
    ],
    "degas": ["#591d06", "#96410e", "#e5a335", "#556219", "#418979", "#2b614e", "#053c29"],
    "nizami": ["#dd7867", "#b83326", "#c8570d", "#edb144", "#8cc8bc", "#7da7ea", "#5773c0", "#1d4497"],
    "gauguin": ["#b04948", "#811e18", "#9e4013", "#c88a2c", "#4c6216", "#1a472a"],
}

#: MoMAColors palettes by name.
MOMA: dict[str, list[str]] = {
    "warhol": [
        "#ff0066",
        "#328c97",
        "#d1aac2",
        "#a5506d",
        "#b3e0bf",
        "#2A9D3D",
        "#edf181",
        "#db7003",
        "#fba600",
        "#f8c1a6",
        "#A30000",
        "#ff3200",
        "#011a51",
        "#97d1d9",
        "#916c37",
    ],
    "klein": [
        "#ff4d6f",
        "#579ea4",
        "#df7713",
        "#f9c000",
        "#86ad34",
        "#5d7298",
        "#81b28d",
        "#7e1a2f",
        "#2d2651",
        "#c8350d",
        "#bd777a",
    ],
    "picasso": ["#d5968c", "#c2676d", "#5c363a", "#995041", "#45939c", "#0f6a81"],
    "koons": ["#d8537d", "#6DC5B2", "#eeca76", "#5d2314", "#b5282a"],
    "dali": [
        "#b4b87f",
        "#9c913f",
        "#585b33",
        "#6ea8ab",
        "#397893",
        "#31333f",
        "#8f5715",
        "#ba9a44",
        "#cfbb83",
    ],
    "andri": ["#f56455", "#15134b", "#87c785", "#572f30"],
    "levine1": ["#E0D9B2", "#818053", "#6B3848", "#8B3E50", "#D5BB6C", "#3F3A4B", "#474C66", "#A5806F"],
    "levine2": ["#E3C1CB", "#AD5A6B", "#C993A2", "#365C83", "#384351", "#4D8F8B", "#CDD6AD"],
    "momavangogh": ["#c3a016", "#c3d878", "#58a787", "#8ebacd", "#246893", "#163274", "#0C1F4b"],
    "ernst": ["#e8e79a", "#c2d89a", "#8cbf9a", "#5fa2a4", "#477b95", "#315b88", "#24396b", "#191f40"],
    "flash": ["#e3c0db", "#db95cb", "#cd64b5", "#B83D9F", "#900c7e", "#680369", "#41045a", "#140e3a"],
}

#: David Hockney, *A Bigger Grand Canyon* (1998): one colour per hue family,
#: ordered most-distinct-first for categorical use.
HOCKNEY_CANYON: list[str] = [
    "#ef5a1c",  # vivid mesa orange
    "#2c9be2",  # cerulean sky blue
    "#d5388a",  # hot magenta (the right-hand mesa)
    "#2fa35c",  # emerald green
    "#f2ab2e",  # golden ochre
    "#9a4fbc",  # violet / purple
    "#d62f1c",  # vermilion red
    "#3fc6dd",  # turquoise (the foreground pools)
    "#b9c233",  # chartreuse lime
    "#a82c55",  # deep plum / rose
    "#f5834b",  # coral
    "#2e7d46",  # pine green
]

#: Stops of the continuous ``"hockney_canyon"`` colormap: indigo, blue,
#: turquoise, green, lime, gold, orange, red, magenta.
HOCKNEY_CANYON_SEQ: list[str] = [
    "#3a1d5c",
    "#2c5fb0",
    "#2c9be2",
    "#3fc6dd",
    "#2fa35c",
    "#b9c233",
    "#f2ab2e",
    "#ef5a1c",
    "#d62f1c",
    "#d5388a",
]

#: The default colour cycle: a bright teal, then the Hockney colours without
#: their turquoise (too close to the teal).
NICE: list[str] = [
    "#25c5a6",  # bright teal (lead)
    "#ef5a1c",  # vivid orange
    "#9a4fbc",  # violet / purple
    "#2c9be2",  # cerulean blue
    "#d5388a",  # hot magenta
    "#f2ab2e",  # golden ochre
    "#2fa35c",  # emerald green
    "#d62f1c",  # vermilion red
    "#b9c233",  # chartreuse lime
    "#a82c55",  # deep plum / rose
    "#f5834b",  # coral
    "#2e7d46",  # pine green
]

#: Names of the individual :data:`NICE` colours, for one-off accents.
NICE_COLORS: dict[str, str] = {
    "teal": NICE[0],
    "orange": NICE[1],
    "purple": NICE[2],
    "blue": NICE[3],
    "magenta": NICE[4],
    "gold": NICE[5],
    "emerald": NICE[6],
    "red": NICE[7],
    "lime": NICE[8],
    "plum": NICE[9],
    "coral": NICE[10],
    "pine": NICE[11],
}


def light_cmap(color: ColorType, name: str = "light") -> mcolors.LinearSegmentedColormap:
    """Single-hue sequential colormap from white through ``color`` to a darker ``color``.

    Suited to density backgrounds (KDEs, hexbins).

    Args:
        color: Any matplotlib colour.
        name: Name of the colormap.
    """
    r, g, b = mcolors.to_rgb(color)
    h, li, s = colorsys.rgb_to_hls(r, g, b)
    dark = colorsys.hls_to_rgb(h, max(li * 0.55, 0.0), s)
    return mcolors.LinearSegmentedColormap.from_list(name, ["#ffffff", color, dark])


#: All qualitative artist palettes by name.
ARTIST_PALETTES: dict[str, list[str]] = {
    **METBREWER,
    **MOMA,
    "hockney": HOCKNEY_CANYON,
    "hockney_canyon": HOCKNEY_CANYON,
    "nice": NICE,
}

# Palettes that read well as continuous colormaps.
_CONTINUOUS = {
    "hockney_canyon": HOCKNEY_CANYON_SEQ,
    "hiroshige": METBREWER["hiroshige"],  # diverging red↔blue
    "hokusai2": METBREWER["hokusai2"],  # sequential blue
    "hokusai3": METBREWER["hokusai3"],  # sequential
    "vangogh3": METBREWER["vangogh3"],  # sequential green
    "nizami": METBREWER["nizami"],  # diverging
    "ernst": MOMA["ernst"],  # sequential
    "flash": MOMA["flash"],  # sequential magenta
}


def artist_palette(name: str, n: int | None = None, *, reverse: bool = False) -> list[str]:
    """Colours of the artist palette ``name``.

    Args:
        name: A key of :data:`ARTIST_PALETTES`, case-insensitive.
        n: Number of colours; the palette repeats if ``n`` exceeds its length.
            ``None`` returns the whole palette.
        reverse: Reverse the order before selecting.

    Returns:
        Hex colour strings.

    Raises:
        KeyError: Unknown palette.
    """
    try:
        cols = list(ARTIST_PALETTES[name.lower()])
    except KeyError as exc:
        raise KeyError(f"Unknown artist palette {name!r}. Available: {sorted(ARTIST_PALETTES)}") from exc
    if reverse:
        cols = cols[::-1]
    if n is None:
        return cols
    if n <= len(cols):
        return cols[:n]
    return [cols[i % len(cols)] for i in range(n)]


def artist_cmap(name: str, *, reverse: bool = False) -> mcolors.LinearSegmentedColormap:
    """Smooth colormap interpolating an artist palette's colours.

    ``"hockney_canyon"`` uses :data:`HOCKNEY_CANYON_SEQ`; other palettes
    interpolate their colours in order.

    Args:
        name: A key of :data:`ARTIST_PALETTES`, case-insensitive.
        reverse: Reverse the colour order.

    Raises:
        KeyError: Unknown palette.
    """
    key = name.lower()
    stops = _CONTINUOUS.get(key) or ARTIST_PALETTES.get(key)
    if stops is None:
        raise KeyError(f"Unknown artist palette {name!r}. Available: {sorted(ARTIST_PALETTES)}")
    if reverse:
        stops = list(stops)[::-1]
    return mcolors.LinearSegmentedColormap.from_list(key, list(stops))


def _register(custom_cmaps: dict[str, mcolors.Colormap]) -> None:
    """Add the continuous artist colormaps and their reverses to ``custom_cmaps``."""
    for key in _CONTINUOUS:
        custom_cmaps.setdefault(key, artist_cmap(key))
        custom_cmaps.setdefault(f"{key}_r", artist_cmap(key, reverse=True))
    for accent in ("teal", "orange", "purple"):
        custom_cmaps.setdefault(f"nice_{accent}", light_cmap(NICE_COLORS[accent], f"nice_{accent}"))
