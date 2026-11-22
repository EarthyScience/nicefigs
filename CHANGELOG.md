# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Concept-figure primitives in `nicefigs.schematic`: `schematic_axes` (a figure of the printed size
  whose data units are inches), `numbered_disc`, `group_band`, `label_pair`, `grid_cells`,
  `folder_icon` and the colour helpers `tint`, `shade` and `soft_color`. They draw with the active
  theme and the `NICE_COLORS` of the house palette, and combine with `diagram_box` and
  `diagram_arrow`. `NICE_COLORS` is now exported at the top level.
- `save(..., check_text=True)` runs `check_text_overlaps` first and raises `ValueError` on a problem.
- Gallery example `35_schematic`.
- The font chain reaches metric-compatible Helvetica clones (TeX Gyre Heros, Nimbus Sans, Liberation Sans) before DejaVu Sans.
- `register_fonts(dir)` (default `$NICEFIGS_FONT_DIR`) makes a folder of fonts available to a theme; the themes' fonts
  are unchanged. SVG text is written as paths (`svg.fonttype="path"` in every theme and in `save`); text
  keeps the list of installed families it was created with, so a figure saved after its theme context
  still uses them.
- `use_agg()`, called by `figure` and `subplots`: selects the Agg backend unless a backend was chosen.
- `scorecard` draws the PhD-defense-deck look: text is the raw value, colour the percent better
  than a baseline on the fixed 11-band `scorecard_skill_cmap` (blue better, red worse).
  New arguments: `color_values` and `text` (colour numbers apart from printed ones),
  `reference_row` (outlined, grey, labelled `baseline_suffix`) and `reference_cells` (one outlined
  cell per column), both also supplying `reference`; `unresolved` (grey cell with grey value),
  `na_text`, `row_groups` and `row_group_gap`, `group_rule`, `cell_in`, `font_scale` and `floor`.
- `scorecard_skill_cmap()` returns the reversed `scorecard_diverging_cmap` pair (higher is better).
- `skill_gap(mode="percent", floor=...)` gives the percent better than the reference.
- Gallery example `34_scorecard_baseline`.

### Changed

- **Behaviour change** of `scorecard` defaults: `highlight_best=False`; with no baseline cells are
  ranked per column on the skill palette (blue best) instead of `RdYlGn`; `reference=` colours by
  percent better (with `uncertainty=` as before, in standard errors, on the same palette); `cmap=None`;
  `group_gap=0.35` (was 0.6); `na_color="#e8e8e8"`; `colorbar_label=None` gives
  "% better than the baseline" (pass `""` for none).
- **Behaviour change**: a new figure is sized from its cells (`cell_in`, 0.42 in) rather than from
  `width="half"`; `width=` still fixes the figure width and solves the cell size. `square` only
  acts on a given `ax`.
- **Behaviour change**: column headings are horizontal and wrapped, units are a grey second line,
  group titles are regular weight above a 1.6 pt rule, row and column labels are text rather than
  ticks, and the colourbar is half the grid width with ticks at the band edges and pointed ends.
- `outline_boxes` draws unfilled rectangles (`fill=False`).

### Notes

- `nicefigs.cmap("RdBu")` returns `RdBu_r` (red high), so `scorecard(reference=..., cmap="RdBu")`
  colours better cells red. Pass `matplotlib.colormaps["RdBu"]` or use the default scale.

## [1.2.0] - 2026-11-20

### Added

- `diagram_arrow(style="line")` draws a flowchart connector: a stroked line with a filled
  head, sized in points; `via` routes an arrow through corner points and `linestyle` dashes it.
- `diagram_box` takes `linestyle` and `hatch`.
- `band` shades a range between two bounds along x, or up y for a profile, with an optional
  legend label; `lines`, `timeseries` and `profile` draw their `bands` with it.

### Changed

- `diagram_box(rounding=0)` draws a plain rectangle, without zero-size corner curves.

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
