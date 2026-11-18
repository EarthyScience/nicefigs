# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-11-18

### Added

- `polygon_mesh` draws cells of any shape (hexagons and pentagons of a grid) from their
  corners, filled by value, by class, by a given colour or not at all.
- `outline_boxes` outlines rectangles in data coordinates, to highlight cells, rows or
  columns of a grid.
- `one_to_one`, `kde_contours`, `scatter_regression` and `histogram`: the 1:1 line, Gaussian-KDE
  contours over paired samples, the target-against-model panel with a regression per model, and a
  histogram that overlays on one axes.
- `colorbar(cax=...)` draws a bar into axes of your own, with the same styling.
- `flux_diagram` draws a box model's budget: a grid of boxes coloured on a diverging
  scale, with sign-aware arrows ∝ √|value| for vertical, horizontal, lateral and ground
  fluxes, an optional split of the vertical flux, and per-box notes; `flux_legend`
  explains it, and `data.box_budget` gives a closed example budget.
- `diagram_box` and `diagram_arrow`, the labelled box and the point-width arrow
  `flux_diagram` is built from, for flowcharts and schematics.
- `profile(bands=...)` shades a range, such as an interquartile range, behind a
  profile in its colour.
- `legend_handle` builds a line, marker or band legend entry in the `role_style`
  vocabulary.

## [1.0.0] - 2026-11-07

First public release.

### Added

- House style: `set_theme`, the scoped `style` context manager and the `Theme`
  presets `paper`, `paper-large`, `talk` and `poster`.
- Physical figure sizing: `figure`, `subplots` and `figure_size` with named
  journal and slide widths; `save` to several formats at once;
  `check_text_overlaps` to find colliding or clipped text.
- Maps: `map`, `scatter_map`, `binned_map`, `raster_map`, `categorical_map`,
  `tri_map`, `hatch_region`, `from_xarray`, with optional cartopy projections
  and an inset-histogram colorbar.
- Colour: continuous and categorical palettes (`cmap`, `categorical`,
  `qualitative`, `elevation_cmap`, land-cover palettes), a role palette
  (`ROLES`, `role_color`, `role_style`), artist palettes, ternary and
  bivariate 2D colormaps with legends.
- Plots: `agreement`, `scatter`, `timeseries`, `lines`, `model_comparison`,
  `taylor`, `scorecard` and `skill_gap`, `pareto`, `caterpillar`, grouped and
  stacked bars, `heatmap`, `image_grid`, `overlay_grid`, `hovmoller`,
  `cross_section`, `profile`, `reliability`, `rank_histogram`,
  `precision_recall`, `spectrum` (eigenvalue decay or cumulative explained
  variance), and the `atlas.component_panel` composite.
- Layout helpers: `panel_label`, `label_panels`, `colorbar`, `facet`,
  `row_labels`, `col_labels`, `reference_line`, metric and unit labels.
- Animation: `animate_maps`, `save_animation`, `render_animation`,
  `stitch_frames` (MP4 via ffmpeg, GIF fallback).
- Deterministic synthetic example data in `nicefigs.data`.
- One calling convention: plotting helpers take `ax=None` and return `ax` or
  `(ax, extra)`; figure helpers return `(fig, axes[, extra])`; decorators take
  the axes first. One unit-argument scheme and one "which way is better"
  vocabulary, `DIRECTIONS`.

[Unreleased]: https://github.com/EarthyScience/nicefigs/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/EarthyScience/nicefigs/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/EarthyScience/nicefigs/releases/tag/v1.0.0
