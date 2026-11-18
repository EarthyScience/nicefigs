"""Polygon meshes: cells of any shape drawn from their corners, filled by value or by category."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.collections import PolyCollection

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from numpy.typing import ArrayLike

from .palettes import cmap as resolve_cmap


def polygon_mesh(
    ax: Axes,
    polygons: Sequence[ArrayLike],
    values: ArrayLike | None = None,
    *,
    categories: Sequence[Any] | None = None,
    colors: Mapping[Any, Any] | None = None,
    facecolors: Sequence[Any] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    facecolor: Any = "none",
    edgecolor: Any = None,
    linewidth: float = 0.4,
    rasterized: bool = False,
    **kwargs: Any,
) -> PolyCollection:
    """Draw polygons, such as hexagonal and pentagonal grid cells, filled by value or by category.

    Cells may have different corner counts. Fill them with a colour scale (``values``),
    with a colour per class (``categories`` and ``colors``), with a colour given per cell
    (``facecolors``), or leave them unfilled (outlines only). The collection is added to ``ax`` and the data limits are updated; on a
    map the caller sets the extent and passes ``transform=``.

    Args:
        ax: Target axes.
        polygons: One ``(corners, 2)`` array of ``(x, y)`` per cell, corners in order around the cell.
        values: One value per cell, mapped through ``cmap`` and the limits.
        categories: One class per cell; needs ``colors``.
        colors: Class to colour, for ``categories``.
        facecolors: One fill colour per cell, for a colouring the caller already computed.
        cmap: Colormap name or object, for ``values``.
        vmin: Lower colour limit; the data minimum if omitted.
        vmax: Upper colour limit; the data maximum if omitted.
        norm: Normalisation; overrides ``vmin`` and ``vmax``.
        facecolor: Fill of every cell when neither ``values`` nor ``categories`` is given.
        edgecolor: Edge colour; ``"face"`` draws each edge in its cell's fill, which hides seams;
            the default is ``"face"`` for filled cells and black for unfilled ones.
        linewidth: Edge width (pt).
        rasterized: Rasterise the collection, to keep a vector figure small.
        **kwargs: Passed to :class:`~matplotlib.collections.PolyCollection` (e.g. ``transform``).

    Returns:
        The collection; for ``values`` it is also the mappable of a colorbar.

    Raises:
        ValueError: If more than one of ``values``, ``categories`` and ``facecolors`` is given, ``categories`` lacks
            ``colors`` or a class has no colour, or a length differs from ``polygons``.
    """
    if sum(arg is not None for arg in (values, categories, facecolors)) > 1:
        raise ValueError("give values, categories or facecolors, not several")
    n = len(polygons)
    if edgecolor is None:
        unfilled = values is None and categories is None and facecolors is None
        edgecolor = "black" if unfilled else "face"
    kw: dict[str, Any] = {
        "edgecolors": edgecolor,
        "linewidths": linewidth,
        "rasterized": rasterized,
        **kwargs,
    }
    if values is not None:
        values = np.asarray(values, dtype=float)
        if values.shape != (n,):
            raise ValueError(f"values has shape {values.shape}, expected ({n},)")
        finite = values[np.isfinite(values)]
        if norm is None and finite.size:
            from matplotlib.colors import Normalize

            norm = Normalize(
                vmin=finite.min() if vmin is None else vmin, vmax=finite.max() if vmax is None else vmax
            )
        coll = PolyCollection(polygons, array=values, cmap=resolve_cmap(cmap), norm=norm, **kw)
    elif facecolors is not None:
        if len(facecolors) != n:
            raise ValueError(f"facecolors has {len(facecolors)} entries, expected {n}")
        coll = PolyCollection(polygons, facecolors=list(facecolors), **kw)
    elif categories is not None:
        if colors is None:
            raise ValueError("categories need colors, a mapping from class to colour")
        if len(categories) != n:
            raise ValueError(f"categories has {len(categories)} entries, expected {n}")
        missing = {c for c in categories if c not in colors}
        if missing:
            raise ValueError(f"no colour for class(es) {sorted(map(str, missing))}")
        coll = PolyCollection(polygons, facecolors=[colors[c] for c in categories], **kw)
    else:
        coll = PolyCollection(polygons, facecolors=facecolor, **kw)
    ax.add_collection(coll)
    ax.autoscale_view()
    return coll
