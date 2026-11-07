"""Global and regional maps of points, gridded fields, unstructured meshes and class labels.

The default map is a Robinson projection with Antarctica cropped, a light land
backdrop, coastlines and an inset histogram that doubles as the colorbar.
:func:`map` picks a drawing method from the data layout; the per-layout helpers
(:func:`scatter_map`, :func:`binned_map`, :func:`raster_map`, :func:`tri_map`,
:func:`categorical_map`) draw onto an existing axes. Without cartopy, maps fall
back to plain equirectangular axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .colorbars import colorbar, inset_histogram_colorbar
from .palettes import cmap as resolve_cmap
from .palettes import robust_limits
from .style import COLORS, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure
    from matplotlib.gridspec import SubplotSpec

#: Global ``(west, east, south, north)`` extent in degrees with Antarctica cropped.
ANTARCTICA_CUTOFF_EXTENT = (-180.0, 180.0, -60.0, 84.0)

#: Above this many points, ``kind="auto"`` bins the cloud instead of scattering it.
AUTO_BIN_MIN_POINTS = 5000

_PROJECTIONS = {
    "robinson": "Robinson",
    "platecarree": "PlateCarree",
    "mollweide": "Mollweide",
    "equalearth": "EqualEarth",
    "orthographic": "Orthographic",
    "nearside": "NearsidePerspective",
}


def _try_cartopy() -> ModuleType | None:
    try:
        import cartopy.crs as ccrs

        return ccrs
    except ImportError:
        return None


def _layer_axes(ax: Axes | None, ccrs: ModuleType | None) -> tuple[Axes, ModuleType | None]:
    """Resolve the axes and ``cartopy.crs`` module a map layer draws with."""
    if ax is None:
        from .figure import figure

        return make_map_axes(figure(width="full", aspect=0.5))
    if ccrs is None and hasattr(ax, "projection"):
        ccrs = _try_cartopy()
    return ax, ccrs


def make_map_axes(
    fig: Figure,
    subplot_spec: int | tuple[int, int, int] | SubplotSpec = 111,
    *,
    projection: str = "robinson",
    central_longitude: float = 0.0,
    **proj_kwargs: Any,
) -> tuple[Axes, ModuleType | None]:
    """Add a map axes to ``fig``, projected when cartopy is installed.

    Args:
        fig: Figure to add the axes to.
        subplot_spec: An int such as ``111``, a ``(nrows, ncols, index)`` tuple,
            or a :class:`~matplotlib.gridspec.SubplotSpec`.
        projection: ``"robinson"``, ``"platecarree"``, ``"mollweide"``,
            ``"equalearth"``, ``"orthographic"`` or ``"nearside"``.
        central_longitude: Central meridian in degrees east, where the
            projection supports it.
        **proj_kwargs: Forwarded to the cartopy projection class.

    Returns:
        ``(ax, ccrs)``: the axes and the ``cartopy.crs`` module, or a plain axes
        and ``None`` without cartopy.

    Raises:
        ValueError: ``projection`` is not one of the names above.
    """
    key = projection.lower()
    if key not in _PROJECTIONS:
        raise ValueError(f"Unknown projection {projection!r}. Known: {sorted(_PROJECTIONS)}")
    ccrs = _try_cartopy()
    proj = None
    if ccrs is not None:
        cls = getattr(ccrs, _PROJECTIONS[key])
        try:
            proj = cls(central_longitude=central_longitude, **proj_kwargs)
        except TypeError:
            proj = cls(**proj_kwargs)
    if isinstance(subplot_spec, tuple):
        ax = fig.add_subplot(*subplot_spec, projection=proj)
    else:
        ax = fig.add_subplot(subplot_spec, projection=proj)
    return ax, ccrs


def draw_background(
    ax: Axes,
    ccrs: ModuleType | None,
    *,
    land: bool = True,
    ocean: bool = False,
    coastline: bool = True,
    borders: bool = False,
    land_color: str = COLORS["land"],
    ocean_color: str = COLORS["ocean"],
    coast_color: str = COLORS["coast"],
    coast_lw: float = 0.3,
    resolution: str = "110m",
) -> None:
    """Draw a land, ocean, coastline and border backdrop on a projected map axes.

    The features are rasterized so vector output (SVG, PDF) stays small while text
    and frame remain vectors. Does nothing when ``ccrs`` is ``None``.

    Args:
        ax: Cartopy ``GeoAxes``, e.g. from :func:`make_map_axes`.
        ccrs: The ``cartopy.crs`` module, or ``None`` for a plain axes.
        land: Fill land with ``land_color``.
        ocean: Fill ocean with ``ocean_color``.
        coastline: Draw coastlines.
        borders: Draw country borders.
        land_color: Land fill colour.
        ocean_color: Ocean fill colour.
        coast_color: Coastline and border colour.
        coast_lw: Coastline width in points; borders are 0.7 times this.
        resolution: Natural Earth coastline resolution: ``"110m"``, ``"50m"`` or ``"10m"``.
    """
    if ccrs is None:
        return
    import cartopy.feature as cfeature

    def _rasterize(artist: Any) -> None:
        if artist is not None and hasattr(artist, "set_rasterized"):
            artist.set_rasterized(True)

    if ocean:
        _rasterize(ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor="none", zorder=0))
    if land:
        _rasterize(ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="none", zorder=0))
    if coastline:
        try:
            _rasterize(
                ax.coastlines(resolution=resolution, linewidth=coast_lw, color=coast_color, zorder=2.5)
            )
        except OSError:  # Natural Earth data unavailable (e.g. offline): skip coastlines.
            pass
    if borders:
        _rasterize(
            ax.add_feature(cfeature.BORDERS, linewidth=coast_lw * 0.7, edgecolor=coast_color, zorder=2.5)
        )
    try:
        ax.spines["geo"].set_edgecolor("#333333")
        ax.spines["geo"].set_linewidth(0.7)
    except (KeyError, AttributeError):
        pass


def _apply_extent(
    ax: Axes, ccrs: ModuleType | None, extent: Sequence[float] | None, antarctica_cutoff: bool
) -> None:
    if ccrs is None:
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        elif antarctica_cutoff:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 84)
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        return
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif antarctica_cutoff:
        ax.set_extent(ANTARCTICA_CUTOFF_EXTENT, crs=ccrs.PlateCarree())
    else:
        ax.set_global()


@lru_cache(maxsize=16)
def _land_mask(nx: int, ny: int, bin_deg: float, resolution: str) -> np.ndarray | None:
    """Boolean land mask at grid-cell centres (cached)."""
    try:
        from cartopy.io.shapereader import Reader, natural_earth
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        return None
    shp = natural_earth(resolution=resolution, category="physical", name="land")
    land = prep(unary_union(list(Reader(shp).geometries())))
    lon_c = -180.0 + bin_deg * (np.arange(nx) + 0.5)
    lat_c = -90.0 + bin_deg * (np.arange(ny) + 0.5)
    lonc, latc = np.meshgrid(lon_c, lat_c)
    mask = np.array([land.contains(Point(x, y)) for x, y in zip(lonc.ravel(), latc.ravel())]).reshape(
        lonc.shape
    )
    return mask


def _grid_edges(
    lon: np.ndarray, lat: np.ndarray, bin_deg: float, global_grid: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``(lon_edges, lat_edges)`` of a ``bin_deg`` grid covering the given points.

    With ``global_grid`` the edges span the whole globe — the origin the land
    mask assumes. Otherwise they snap to the points' bounding box, aligned to
    that same global grid, so regional maps don't build a whole-globe fine mesh.
    """
    if global_grid:
        return (
            np.arange(-180.0, 180.0 + bin_deg, bin_deg),
            np.arange(-90.0, 90.0 + bin_deg, bin_deg),
        )

    def _edges(lo: float, hi: float, gmin: float, gmax: float) -> np.ndarray:
        e0 = np.floor((lo - gmin) / bin_deg) * bin_deg + gmin
        e1 = np.ceil((hi - gmin) / bin_deg) * bin_deg + gmin
        return np.arange(max(e0, gmin), min(e1, gmax) + bin_deg, bin_deg)

    return _edges(lon.min(), lon.max(), -180.0, 180.0), _edges(lat.min(), lat.max(), -90.0, 90.0)


def _bin_values(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    bin_deg: float,
    fill_bins: int,
    clip_ocean: bool,
    land_resolution: str,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    finite = np.isfinite(values)
    lon_edges, lat_edges = _grid_edges(lon[finite], lat[finite], bin_deg, clip_ocean)
    s, _, _ = np.histogram2d(lat[finite], lon[finite], bins=(lat_edges, lon_edges), weights=values[finite])
    c, _, _ = np.histogram2d(lat[finite], lon[finite], bins=(lat_edges, lon_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(c > 0, s / c, np.nan)
    if fill_bins > 0:
        from scipy.interpolate import griddata
        from scipy.ndimage import distance_transform_edt

        filled = np.isfinite(grid)
        if filled.any():
            yy, xx = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
            interp = griddata(
                np.column_stack([yy[filled], xx[filled]]), grid[filled], (yy, xx), method="linear"
            )
            dist = distance_transform_edt(~filled)
            fill = (dist <= float(fill_bins)) & np.isfinite(interp) & ~filled
            grid = np.where(fill, interp, grid)
    if clip_ocean:
        mask = _land_mask(grid.shape[1], grid.shape[0], float(bin_deg), land_resolution)
        if mask is not None:
            grid = np.where(mask, grid, np.nan)
    return lon_edges, lat_edges, np.ma.masked_invalid(grid)


def hatch_region(
    ax: Axes,
    lon: ArrayLike,
    lat: ArrayLike,
    mask: ArrayLike,
    *,
    ccrs: ModuleType | None = None,
    hatch: str = "////",
    edgecolor: str = "#333333",
    linewidth: float = 0.5,
    bin_deg: float = 2.0,
    threshold: float = 0.5,
) -> QuadContourSet:
    """Hatch the region where a per-point mask is set, leaving the colours beneath visible.

    The mask is bin-averaged onto a ``bin_deg`` grid and cells at or above
    ``threshold`` are hatched without fill.

    Args:
        ax: Map axes.
        lon: Point longitudes in degrees east.
        lat: Point latitudes in degrees north.
        mask: Per-point boolean or fraction, same shape as ``lon``.
        ccrs: The ``cartopy.crs`` module when ``ax`` is projected, else ``None``.
        hatch: Matplotlib hatch pattern.
        edgecolor: Hatch colour.
        linewidth: Hatch line width in points.
        bin_deg: Grid cell size in degrees.
        threshold: Minimum cell-mean mask value that is hatched.

    Returns:
        The contour set drawing the hatch.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    m = np.asarray(mask, dtype=float)
    lon_e, lat_e, grid = _bin_values(lon, lat, m, bin_deg, 0, False, "110m")
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])
    filled = np.ma.filled(grid, 0.0)
    kw = {}
    if ccrs is not None:
        kw["transform"] = ccrs.PlateCarree()
    cs = ax.contourf(lon_c, lat_c, filled, levels=[threshold, 1e9], colors="none", hatches=[hatch], **kw)
    # matplotlib < 3.8 exposes per-level ``.collections``; later the ContourSet is the collection.
    targets = getattr(cs, "collections", None) or [cs]
    for coll in targets:
        coll.set_edgecolor(edgecolor)
        coll.set_linewidth(linewidth)
    return cs


def scatter_map(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    ccrs: ModuleType | None = None,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    size: float = 6.0,
    alpha: float = 0.9,
    sort_by_value: bool = True,
    **scatter_kw: Any,
) -> tuple[Axes, PathCollection]:
    """Scatter points on a map, coloured by ``values`` if given.

    Points with non-finite values are dropped.

    Args:
        ax: Map axes; a Robinson map figure is created if ``None``.
        lon: Longitudes in degrees east, shape ``(N,)``.
        lat: Latitudes in degrees north, shape ``(N,)``.
        values: Per-point values; ``None`` draws plain locations.
        ccrs: The ``cartopy.crs`` module; detected from ``ax`` if ``None``.
        cmap: Colormap name or object.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        size: Marker area in points².
        alpha: Marker opacity.
        sort_by_value: Draw low values first so high values sit on top.
        **scatter_kw: Forwarded to :meth:`~matplotlib.axes.Axes.scatter`.

    Returns:
        ``(ax, mappable)``: the axes and the scatter collection (a mappable when ``values`` is given).
    """
    ax, ccrs = _layer_axes(ax, ccrs)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    kw = dict(s=size, linewidths=0, alpha=alpha, rasterized=True)
    if ccrs is not None:
        kw["transform"] = ccrs.PlateCarree()
    if values is None:
        return ax, ax.scatter(lon, lat, **kw, **scatter_kw)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    lon, lat, values = lon[valid], lat[valid], values[valid]
    if sort_by_value:
        order = np.argsort(values)
        lon, lat, values = lon[order], lat[order], values[order]
    kw.update(c=values, cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax)
    if norm is not None:
        kw["norm"] = norm
        kw.pop("vmin")
        kw.pop("vmax")
    return ax, ax.scatter(lon, lat, **kw, **scatter_kw)


def binned_map(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    ccrs: ModuleType | None = None,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    bin_deg: float = 2.0,
    fill_bins: int = 0,
    clip_ocean: bool = False,
    land_resolution: str = "110m",
    **mesh_kw: Any,
) -> tuple[Axes, QuadMesh]:
    """Average point values onto a regular lon/lat grid and draw it as a mesh.

    Args:
        ax: Map axes; a Robinson map figure is created if ``None``.
        lon: Point longitudes in degrees east, shape ``(N,)``.
        lat: Point latitudes in degrees north, shape ``(N,)``.
        values: Point values, shape ``(N,)``; non-finite values are ignored.
        ccrs: The ``cartopy.crs`` module; detected from ``ax`` if ``None``.
        cmap: Colormap name or object.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        bin_deg: Grid cell size in degrees.
        fill_bins: Fill empty cells within this many cells of data by linear
            interpolation (requires scipy); 0 disables filling.
        clip_ocean: Mask cells whose centre lies at sea (requires cartopy and
            shapely); the grid then spans the globe.
        land_resolution: Natural Earth resolution of the land mask.
        **mesh_kw: Forwarded to :meth:`~matplotlib.axes.Axes.pcolormesh`.

    Returns:
        ``(ax, mappable)``: the axes and the ``QuadMesh`` mappable.
    """
    ax, ccrs = _layer_axes(ax, ccrs)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    values = np.asarray(values, dtype=float)
    lon_e, lat_e, grid = _bin_values(lon, lat, values, bin_deg, fill_bins, clip_ocean, land_resolution)
    kw = dict(cmap=resolve_cmap(cmap), shading="flat", rasterized=True)
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if ccrs is not None:
        kw["transform"] = ccrs.PlateCarree()
    return ax, ax.pcolormesh(lon_e, lat_e, grid, **kw, **mesh_kw)


def _resolve_categories(
    categories: Mapping[Any, str] | Sequence[str], codes: ArrayLike
) -> tuple[list, list[str]]:
    """Normalise ``categories`` to parallel ``(codes, names)`` lists.

    A mapping gives ``code -> name`` in its own order; a sequence of names is
    read as codes ``0..n-1``, or — for string labels — as the codes themselves.
    """
    if hasattr(categories, "items"):
        items = list(categories.items())
        if not items:
            raise ValueError("categories must not be empty")
        return [c for c, _ in items], [str(n) for _, n in items]
    names = [str(n) for n in categories]
    if not names:
        raise ValueError("categories must not be empty")
    if np.asarray(codes).dtype.kind in "USO":
        return list(names), names
    return list(range(len(names))), names


def _resolve_category_colors(
    colors: Mapping[Any, str] | Sequence[str] | None, codes: list, names: list[str]
) -> list:
    """Resolve ``colors`` to one colour per category, defaulting to the house palette."""
    if colors is None:
        from .palettes import qualitative

        return qualitative(len(codes))
    if hasattr(colors, "items"):
        missing = [n for n, c in zip(names, codes) if n not in colors and c not in colors]
        if missing:
            raise ValueError(f"colors is missing an entry for categories {missing}")
        return [colors[n] if n in colors else colors[c] for n, c in zip(names, codes)]
    colors = list(colors)
    if len(colors) < len(codes):
        raise ValueError(f"colors has {len(colors)} entries for {len(codes)} categories")
    return colors[: len(codes)]


def _category_indices(codes: ArrayLike, cat_codes: list) -> np.ndarray:
    """Index of each point's category, ``-1`` where the code is unknown or missing."""
    arr = np.asarray(codes)
    index = np.full(arr.shape, -1, dtype=int)
    for i, code in enumerate(cat_codes):
        # A dtype-mismatched comparison (string array vs numeric code, say)
        # yields a scalar instead of an elementwise mask: no point matches.
        hit = np.asarray(arr == code)
        if hit.shape == arr.shape:
            index[hit] = i
    return index


def _bin_majority(
    lon: np.ndarray, lat: np.ndarray, index: np.ndarray, n_categories: int, bin_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Majority-vote ``index`` onto a ``bin_deg`` grid; empty cells are masked."""
    lon_edges, lat_edges = _grid_edges(lon, lat, bin_deg, False)
    counts = np.stack(
        [
            np.histogram2d(lat[index == i], lon[index == i], bins=(lat_edges, lon_edges))[0]
            for i in range(n_categories)
        ]
    )
    total = counts.sum(axis=0)
    # argmax breaks ties towards the earlier category, i.e. the caller's order.
    winner = np.argmax(counts, axis=0)
    return lon_edges, lat_edges, np.ma.masked_where(total == 0, winner)


def categorical_map(
    lon: ArrayLike,
    lat: ArrayLike,
    codes: ArrayLike,
    *,
    ax: Axes | None = None,
    categories: Mapping[Any, str] | Sequence[str],
    ccrs: ModuleType | None = None,
    colors: Mapping[Any, str] | Sequence[str] | None = None,
    kind: Literal["auto", "binned", "scatter"] = "auto",
    bin_deg: float = 2.0,
    legend: bool = True,
    legend_kw: dict[str, Any] | None = None,
    **kw: Any,
) -> tuple[Axes, QuadMesh | PathCollection]:
    """Map per-point class labels as discrete colours with a matching legend.

    Class codes cannot be averaged, so dense clouds are reduced to a ``bin_deg``
    grid by **majority vote** (ties go to the earlier category) rather than by
    the mean :func:`binned_map` takes; sparse clouds are drawn as a scatter.
    Points whose code is not in ``categories``, and points with a non-finite
    coordinate, are dropped.

    Args:
        ax: Map axes; a Robinson map figure is created if ``None``.
        lon: Point longitudes in degrees east.
        lat: Point latitudes in degrees north.
        codes: Per-point class label (integer or string), same shape as ``lon``.
        categories: Ordered class names — a ``{code: name}`` mapping for
            arbitrary codes, or a sequence of names read as codes ``0..n-1``
            (or, for string labels, as the codes themselves).
        ccrs: The ``cartopy.crs`` module; detected from ``ax`` if ``None``.
        colors: One colour per category, as a sequence in category order or a
            mapping keyed by category name or code. Defaults to
            :func:`nicefigs.qualitative`.
        kind: ``"auto"`` (binned above 5000 valid points, else scatter),
            ``"binned"``, or ``"scatter"``.
        bin_deg: Grid cell size in degrees for the binned path.
        legend: Draw a discrete patch legend, one entry per category present.
        legend_kw: Overrides forwarded to :meth:`~matplotlib.axes.Axes.legend`.
        **kw: Forwarded to :meth:`~matplotlib.axes.Axes.pcolormesh` or :func:`scatter_map`.

    Returns:
        ``(ax, mappable)``: the axes and the mappable (a ``QuadMesh`` or ``PathCollection``). The legend, when
        drawn, is reachable as ``ax.get_legend()``.

    Example:
        >>> categorical_map(lon, lat, codes, categories=["in", "out"], ax=ax)  # doctest: +SKIP
    """
    ax, ccrs = _layer_axes(ax, ccrs)
    import matplotlib.colors as mcolors

    cat_codes, cat_names = _resolve_categories(categories, codes)
    cat_colors = _resolve_category_colors(colors, cat_codes, cat_names)
    n = len(cat_codes)
    cmap = mcolors.ListedColormap(cat_colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), n)

    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    index = _category_indices(codes, cat_codes).ravel()
    valid = (index >= 0) & np.isfinite(lon) & np.isfinite(lat)
    lon, lat, index = lon[valid], lat[valid], index[valid]

    binned = kind == "binned" or (kind == "auto" and index.size > AUTO_BIN_MIN_POINTS)
    if binned and index.size:
        lon_e, lat_e, grid = _bin_majority(lon, lat, index, n, bin_deg)
        mesh_kw = dict(cmap=cmap, norm=norm, shading="flat", rasterized=True)
        if ccrs is not None:
            mesh_kw["transform"] = ccrs.PlateCarree()
        mappable = ax.pcolormesh(lon_e, lat_e, grid, **mesh_kw, **kw)
        present = np.unique(np.ma.compressed(grid))
    else:
        _, mappable = scatter_map(
            lon,
            lat,
            index.astype(float),
            ccrs=ccrs,
            cmap=cmap,
            norm=norm,
            sort_by_value=False,
            **kw,
            ax=ax,
        )
        present = np.unique(index)

    if legend and present.size:
        from matplotlib.patches import Patch

        th = get_theme()
        handles = [
            Patch(facecolor=cat_colors[i], edgecolor="none", label=cat_names[i]) for i in present.astype(int)
        ]
        # A class map fills its frame, so the legend sits on a translucent
        # backing rather than over bare colour.
        lkw = dict(
            loc="lower left",
            fontsize=th.fs_legend,
            frameon=True,
            framealpha=0.85,
            facecolor="white",
            edgecolor="none",
            handlelength=1.0,
            handletextpad=0.5,
            borderpad=0.4,
            labelspacing=0.35,
        )
        lkw.update(legend_kw or {})
        ax.legend(handles=handles, **lkw)
    return ax, mappable


def raster_map(
    lon: ArrayLike,
    lat: ArrayLike,
    field: ArrayLike,
    *,
    ax: Axes | None = None,
    ccrs: ModuleType | None = None,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    method: Literal["pcolormesh", "imshow"] = "pcolormesh",
    shading: str = "auto",
    **kw: Any,
) -> tuple[Axes, ScalarMappable]:
    """Draw an already-gridded 2-D field on a map.

    Args:
        ax: Map axes; a Robinson map figure is created if ``None``.
        lon: Cell-centre longitudes in degrees east: 1-D ``(nx,)`` for a regular
            grid, or 2-D ``(ny, nx)`` for a curvilinear one.
        lat: Cell-centre latitudes in degrees north: 1-D ``(ny,)`` or 2-D ``(ny, nx)``.
        field: Values of shape ``(ny, nx)``; NaNs render transparent.
        ccrs: The ``cartopy.crs`` module; detected from ``ax`` if ``None``.
        cmap: Colormap name or object.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        method: ``"pcolormesh"`` handles irregular and curvilinear grids;
            ``"imshow"`` is faster but needs a regular grid with 1-D coordinates.
        shading: ``pcolormesh`` shading mode.
        **kw: Forwarded to :meth:`~matplotlib.axes.Axes.pcolormesh` or
            :meth:`~matplotlib.axes.Axes.imshow`.

    Returns:
        ``(ax, mappable)``: the axes and the ``QuadMesh`` or ``AxesImage`` mappable.
    """
    ax, ccrs = _layer_axes(ax, ccrs)
    field = np.ma.masked_invalid(np.asarray(field, dtype=float))
    kw = dict(kw, cmap=resolve_cmap(cmap), rasterized=True)
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if ccrs is not None:
        kw["transform"] = ccrs.PlateCarree()
    if method == "imshow":
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        origin = "upper" if lat[0] > lat[-1] else "lower"
        return ax, ax.imshow(field, extent=extent, origin=origin, aspect="auto", **kw)
    return ax, ax.pcolormesh(np.asarray(lon), np.asarray(lat), field, shading=shading, **kw)


def tri_map(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    ccrs: ModuleType | None = None,
    cmap: str | Colormap = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    method: Literal["tricontourf", "tripcolor"] = "tricontourf",
    levels: int | Sequence[float] = 14,
    max_edge_frac: float = 0.12,
    **kw: Any,
) -> tuple[Axes, ScalarMappable]:
    """Draw unstructured data (e.g. an icosahedral mesh) by triangulating its points.

    The triangulation is built in projected coordinates, and triangles whose
    longest edge exceeds ``max_edge_frac`` of the map width are masked, so no
    triangle smears across the antimeridian or the poles.

    Args:
        ax: Map axes; a Robinson map figure is created if ``None``.
        lon: Node longitudes in degrees east, shape ``(N,)``.
        lat: Node latitudes in degrees north, shape ``(N,)``.
        values: Node values, shape ``(N,)``; non-finite nodes are dropped.
        ccrs: The ``cartopy.crs`` module; detected from ``ax`` if ``None``.
        cmap: Colormap name or object.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides ``vmin``/``vmax``.
        method: ``"tricontourf"`` for smooth filled contours, ``"tripcolor"`` for
            Gouraud-shaded triangles.
        levels: Contour levels (count or boundaries) for ``tricontourf``.
        max_edge_frac: Mask triangles with an edge longer than this fraction of
            the projected map width.
        **kw: Forwarded to :meth:`~matplotlib.axes.Axes.tricontourf` or
            :meth:`~matplotlib.axes.Axes.tripcolor`.

    Returns:
        ``(ax, mappable)``: the axes and the contour set or ``PolyCollection`` mappable.
    """
    ax, ccrs = _layer_axes(ax, ccrs)
    from matplotlib.tri import Triangulation

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
    lon, lat, values = lon[finite], lat[finite], values[finite]

    # Points are projected here, so the plot call needs no ``transform``.
    if ccrs is not None:
        pts = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)
        x, y = pts[:, 0], pts[:, 1]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, values = x[ok], y[ok], values[ok]
    else:
        x, y = lon, lat

    tri = Triangulation(x, y)
    xt, yt = x[tri.triangles], y[tri.triangles]
    edges = np.sqrt((xt - np.roll(xt, 1, 1)) ** 2 + (yt - np.roll(yt, 1, 1)) ** 2).max(1)
    width = np.nanmax(x) - np.nanmin(x)
    tri.set_mask(edges > max_edge_frac * width)

    kw = dict(kw, cmap=resolve_cmap(cmap))
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if method == "tripcolor":
        return ax, ax.tripcolor(tri, values, shading="gouraud", rasterized=True, **kw)
    return ax, ax.tricontourf(tri, values, levels=levels, **kw)


_LON_NAMES = ("lon", "longitude", "x", "nav_lon", "xc", "clon")
_LAT_NAMES = ("lat", "latitude", "y", "nav_lat", "yc", "clat")


def _is_dataarray(obj: object) -> bool:
    return obj.__class__.__name__ == "DataArray" and hasattr(obj, "dims")


def from_xarray(da: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(lon, lat, field)`` numpy arrays from an xarray ``DataArray``.

    Longitude and latitude coordinates are found by common names (``lon``,
    ``longitude``, ``x``, ``nav_lon``, ...; case-insensitive) after squeezing
    singleton dimensions. Works for regular grids and for unstructured data with
    one cell dimension.

    Args:
        da: An ``xarray.DataArray``.

    Returns:
        ``(lon, lat, field)`` as numpy arrays.

    Raises:
        KeyError: No longitude or latitude coordinate is found.
    """
    da = da.squeeze()
    coords = {str(k).lower(): k for k in da.coords}
    lon_key = next((coords[n] for n in _LON_NAMES if n in coords), None)
    lat_key = next((coords[n] for n in _LAT_NAMES if n in coords), None)
    if lon_key is None or lat_key is None:
        raise KeyError(
            f"Could not find lon/lat coords in {list(da.coords)}. "
            "Pass lon/lat explicitly, or rename coords to lon/lat."
        )
    return np.asarray(da[lon_key].values), np.asarray(da[lat_key].values), np.asarray(da.values)


def map(
    lon: ArrayLike,
    lat: ArrayLike | None = None,
    values: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    fig: Figure | None = None,
    subplot_spec: int | tuple[int, int, int] | SubplotSpec = 111,
    kind: str = "auto",
    projection: str = "robinson",
    central_longitude: float = 0.0,
    extent: Sequence[float] | None = None,
    antarctica_cutoff: bool = True,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    robust: bool = True,
    diverging: bool = False,
    center: float = 0.0,
    bin_deg: float = 2.0,
    fill_bins: int = 0,
    clip_ocean: bool | None = None,
    land: bool = True,
    ocean: bool = False,
    coastline: bool = True,
    borders: bool = False,
    inset_hist: bool = True,
    colorbar_label: str = "",
    cbar: bool = False,
    cbar_kwargs: dict[str, Any] | None = None,
    hist_bounds: tuple[float, float, float, float] = (0.07, 0.22, 0.18, 0.11),
    width: str | float = "full",
    aspect: float = 0.52,
    title: str = "",
    **plot_kw: Any,
) -> tuple[Figure, Axes, ScalarMappable | None]:
    """Draw a global or regional map, choosing the drawing method from the data layout.

    Four layouts are supported:

    * **points**: 1-D ``lon``, ``lat``, ``values``, scattered, or bin-averaged for
      dense clouds;
    * **gridded**: 1-D or 2-D ``lon``/``lat`` with a 2-D ``values`` field, or an
      xarray ``DataArray`` as the only argument (``map(da)``);
    * **unstructured**: 1-D ``lon``/``lat``/``values`` with ``kind="tri"``;
    * **classes**: class labels with ``kind="categorical"`` and ``categories=...``,
      decoded by a patch legend instead of a colorbar.

    Args:
        lon: Longitudes in degrees east, or an xarray ``DataArray`` passed alone.
        lat: Latitudes in degrees north.
        values: Per-point values (1-D) or a gridded field (2-D); ``None`` draws
            just the locations.
        ax: Existing map axes to draw into; created if ``None``.
        fig: Figure to add the axes to when ``ax`` is ``None``.
        subplot_spec: Position of the new axes, as in :func:`make_map_axes`.
        kind: ``"auto"`` (raster for 2-D fields, binned above
            :data:`AUTO_BIN_MIN_POINTS` points, else scatter), ``"scatter"``,
            ``"binned"``, ``"raster"``, ``"imshow"``, ``"tri"``, ``"tripcolor"``
            or ``"categorical"`` (see :func:`categorical_map`).
        projection: Projection name, as in :func:`make_map_axes`.
        central_longitude: Central meridian in degrees east.
        extent: ``(west, east, south, north)`` in degrees for a regional map;
            overrides ``antarctica_cutoff``.
        antarctica_cutoff: Crop the global view to latitudes -60 to 84.
        cmap: Colormap name or object; defaults to ``"RdBu_r"`` when
            ``diverging``, else ``"inferno"``.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        norm: Colour normalisation; overrides the limits.
        robust: Use 2nd/98th-percentile limits when ``vmin``/``vmax`` are unset.
        diverging: Use limits symmetric about ``center`` and a diverging colormap.
        center: Centre value of the diverging normalisation.
        bin_deg: Grid cell size in degrees for binned and categorical maps.
        fill_bins: Gap filling for binned maps, as in :func:`binned_map`.
        clip_ocean: For binned maps, mask cells whose centre is at sea. ``None``
            clips global maps that draw land; pass ``False`` for data measured
            over the ocean.
        land: Draw the land background.
        ocean: Fill the ocean.
        coastline: Draw coastlines.
        borders: Draw country borders.
        inset_hist: Draw the inset histogram that doubles as the colorbar.
        colorbar_label: Label of the histogram and/or colorbar.
        cbar: Also draw a conventional colorbar.
        cbar_kwargs: Keywords for :func:`nicefigs.colorbar`.
        hist_bounds: Inset histogram ``(x0, y0, width, height)`` in axes fractions.
        width: Width of a new figure, a named width or inches.
        aspect: Height / width of a new figure.
        title: Left-aligned panel title.
        **plot_kw: Forwarded to the layout helper (:func:`scatter_map`,
            :func:`binned_map`, :func:`raster_map`, :func:`tri_map` or
            :func:`categorical_map`).

    Returns:
        ``(fig, ax, mappable)``; ``mappable`` is ``None`` when ``values`` is ``None``.
    """
    th = get_theme()
    # An xarray DataArray may be passed as the sole positional argument.
    if _is_dataarray(lon) and lat is None:
        lon, lat, values = from_xarray(lon)
        if kind == "auto":
            kind = "raster"

    if ax is None:
        if fig is None:
            from .figure import figure_size

            fig = plt.figure(figsize=figure_size(width, aspect=aspect), dpi=th.dpi)
        ax, ccrs = make_map_axes(
            fig, subplot_spec, projection=projection, central_longitude=central_longitude
        )
    else:
        fig = ax.figure
        ccrs = _try_cartopy() if hasattr(ax, "projection") else None

    if cmap is None:
        cmap = "RdBu_r" if diverging else "inferno"

    # limits (meaningless for class labels, which carry their own discrete norm)
    if values is not None and norm is None and kind != "categorical":
        if diverging:
            from .palettes import diverging_norm, symmetric_limits

            lo, hi = symmetric_limits(values) if (vmin is None or vmax is None) else (vmin, vmax)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
            norm = diverging_norm(vmin, vmax, center)
        elif robust and (vmin is None or vmax is None):
            lo, hi = robust_limits(values)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    draw_background(ax, ccrs, land=land, ocean=ocean, coastline=coastline, borders=borders)
    _apply_extent(ax, ccrs, extent, antarctica_cutoff and extent is None)

    mappable = None
    if values is not None:
        eff = kind
        if kind == "auto":
            eff = (
                "raster"
                if np.ndim(values) == 2
                else ("binned" if np.size(values) > AUTO_BIN_MIN_POINTS else "scatter")
            )

        if eff in ("raster", "imshow", "pcolormesh"):
            _, mappable = raster_map(
                lon,
                lat,
                values,
                ccrs=ccrs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                method="imshow" if eff == "imshow" else "pcolormesh",
                **plot_kw,
                ax=ax,
            )
        elif eff in ("tri", "tricontourf", "tripcolor"):
            _, mappable = tri_map(
                lon,
                lat,
                values,
                ccrs=ccrs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                method="tripcolor" if eff == "tripcolor" else "tricontourf",
                **plot_kw,
                ax=ax,
            )
        elif eff == "categorical":
            _, mappable = categorical_map(lon, lat, values, ccrs=ccrs, bin_deg=bin_deg, **plot_kw, ax=ax)
        elif eff == "binned":
            # Ocean-clip only for global maps: over a fine regional grid the
            # point-in-polygon land mask is prohibitively slow and rarely wanted.
            clip = (land and extent is None) if clip_ocean is None else bool(clip_ocean)
            _, mappable = binned_map(
                lon,
                lat,
                values,
                ccrs=ccrs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                bin_deg=bin_deg,
                fill_bins=fill_bins,
                clip_ocean=clip,
                **plot_kw,
                ax=ax,
            )
        else:
            _, mappable = scatter_map(
                lon,
                lat,
                values,
                ccrs=ccrs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                **plot_kw,
                ax=ax,
            )
    else:
        scatter_map(lon, lat, ccrs=ccrs, color=COLORS["primary"], **plot_kw, ax=ax)

    if title:
        from .panels import clean_title

        clean_title(ax, title)

    # A categorical map decodes itself through its legend, never a colorbar.
    if mappable is not None and kind != "categorical":
        if inset_hist:
            inset_histogram_colorbar(
                ax,
                values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                bounds=hist_bounds,
                label=colorbar_label or None,
            )
        if cbar:
            cbar_kwargs = dict(cbar_kwargs or {})
            cbar_kwargs.setdefault("label", colorbar_label)
            colorbar(mappable, ax, **cbar_kwargs)

    return fig, ax, mappable
