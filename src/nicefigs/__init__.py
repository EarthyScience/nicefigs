"""nicefigs — publication-grade, modular plotting for geoscience.

Quickstart::

    import nicefigs as nf

    nf.set_theme("paper")                      # house style, globally
    fig, ax, m = nf.map(lon, lat, values)      # Robinson map + inset histogram
    nf.save(fig, "figures/map", formats=("pdf", "png"))

Every helper takes an existing ``ax`` where it makes sense, so you can compose
nicefigs pieces into your own matplotlib layouts. The :mod:`nicefigs.data`
submodule provides deterministic example data for the gallery, docs, and tests.
"""

from __future__ import annotations

from . import data
from .animation import animate_maps, render_animation, save_animation, stitch_frames
from .artists import ARTIST_PALETTES, HOCKNEY_CANYON, NICE, artist_cmap, artist_palette
from .bars import grouped as grouped_bars
from .bars import stacked as stacked_bars
from .colorbars import colorbar, inset_histogram_colorbar, make_mappable
from .colormaps2d import (
    bivariate_cmap,
    bivariate_legend,
    flat_cmap,
    flat_legend,
    hsv_cmap,
    hsv_legend,
)
from .comparison import model_comparison
from .figure import TextIssue, check_text_overlaps, figure, figure_size, save, subplots
from .hovmoller import hovmoller, hovmoller_panels
from .images import heatmap, image, image_grid, overlay_grid, pca_rgb
from .kde import agreement
from .lines import lines
from .maps import (
    binned_map,
    categorical_map,
    draw_background,
    from_xarray,
    hatch_region,
    make_map_axes,
    map,
    raster_map,
    scatter_map,
    tri_map,
)
from .palettes import (
    CATEGORICAL,
    CUSTOM_CMAPS,
    DEFAULTS,
    ROLES,
    SCORECARD_DIVERGING_BOUNDARIES,
    Categorical,
    Role,
    categorical,
    cmap,
    diverging_norm,
    elevation_cmap,
    qualitative,
    robust_limits,
    role_color,
    role_style,
    scorecard_diverging_cmap,
    symmetric_limits,
)
from .panels import (
    clean_title,
    col_labels,
    despine,
    facet,
    grid,
    label_panels,
    panel_label,
    reference_line,
    row_labels,
)
from .pareto import frontier, pareto
from .ranking import caterpillar
from .scatter import scatter
from .sections import cross_section, cross_section_panels, profile
from .spectra import spectrum
from .style import (
    COLORS,
    PALETTE_BRIGHT,
    PALETTE_HIGH_CONTRAST,
    PALETTE_MUTED,
    THEMES,
    Theme,
    get_theme,
    set_theme,
    style,
)
from .tables import scorecard, skill_gap
from .taylor import taylor
from .ternary import (
    TERNARY_PALETTES,
    ternary_axes,
    ternary_image,
    ternary_legend,
    ternary_map,
    ternary_scatter,
)
from .ternary import color as ternary_color
from .timeseries import timeseries
from .units import (
    DIRECTIONS,
    METRIC_DIRECTION,
    METRIC_NAMES,
    axis_label,
    fmt_pval,
    fmt_val,
    metric_label,
    prettify_unit,
    set_labels,
)
from .verification import (
    precision_recall,
    rank_histogram,
    rank_histogram_panels,
    reliability,
)

__version__ = "1.0.0"

__all__ = [
    "data",
    # style
    "set_theme",
    "style",
    "Theme",
    "THEMES",
    "get_theme",
    "COLORS",
    "PALETTE_BRIGHT",
    "PALETTE_MUTED",
    "PALETTE_HIGH_CONTRAST",
    # figure
    "figure",
    "subplots",
    "figure_size",
    "save",
    "check_text_overlaps",
    "TextIssue",
    # panels
    "grid",
    "panel_label",
    "label_panels",
    "clean_title",
    "despine",
    "reference_line",
    "facet",
    "row_labels",
    "col_labels",
    # palettes
    "cmap",
    "categorical",
    "Categorical",
    "CATEGORICAL",
    "DEFAULTS",
    "elevation_cmap",
    "qualitative",
    "diverging_norm",
    "robust_limits",
    "symmetric_limits",
    "scorecard_diverging_cmap",
    "SCORECARD_DIVERGING_BOUNDARIES",
    # role palette
    "ROLES",
    "Role",
    "role_color",
    "role_style",
    # artist palettes
    "artist_palette",
    "artist_cmap",
    "ARTIST_PALETTES",
    "HOCKNEY_CANYON",
    "NICE",
    # colorbars
    "colorbar",
    "inset_histogram_colorbar",
    "make_mappable",
    # units
    "axis_label",
    "metric_label",
    "set_labels",
    "prettify_unit",
    "fmt_val",
    "fmt_pval",
    "METRIC_NAMES",
    "METRIC_DIRECTION",
    # maps
    "map",
    "scatter_map",
    "binned_map",
    "categorical_map",
    "raster_map",
    "tri_map",
    "make_map_axes",
    "draw_background",
    "from_xarray",
    # images
    "image",
    "heatmap",
    "overlay_grid",
    "image_grid",
    "pca_rgb",
    # relational
    "scatter",
    "agreement",
    "timeseries",
    "lines",
    "spectrum",
    "model_comparison",
    "pareto",
    "frontier",
    "caterpillar",
    "grouped_bars",
    "stacked_bars",
    "scorecard",
    "skill_gap",
    "DIRECTIONS",
    "cross_section",
    "cross_section_panels",
    "profile",
    "hatch_region",
    "taylor",
    "hovmoller",
    "hovmoller_panels",
    # verification / generative-model evaluation
    "reliability",
    "rank_histogram",
    "rank_histogram_panels",
    "precision_recall",
    # 2D colormaps
    "bivariate_cmap",
    "hsv_cmap",
    "flat_cmap",
    "bivariate_legend",
    "hsv_legend",
    "flat_legend",
    "CUSTOM_CMAPS",
    # ternary
    "ternary_scatter",
    "ternary_map",
    "ternary_image",
    "ternary_legend",
    "ternary_axes",
    "ternary_color",
    "TERNARY_PALETTES",
    # animation
    "animate_maps",
    "save_animation",
    "render_animation",
    "stitch_frames",
    "__version__",
]
