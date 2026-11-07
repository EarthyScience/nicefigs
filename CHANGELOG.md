# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/EarthyScience/nicefigs/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/EarthyScience/nicefigs/releases/tag/v1.0.0
