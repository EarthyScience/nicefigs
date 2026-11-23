"""Offline Natural Earth data for the map functions.

nicefigs bundles the Natural Earth 110m and 50m coastline, land, ocean and land-border
shapefiles (public domain, naturalearthdata.com) so maps render without internet access,
e.g. on compute nodes. Cartopy looks up data in this order, and nicefigs adds its own
sources only as the last step before a download:

1. ``cartopy.config["pre_existing_data_dir"]`` (``$CARTOPY_DATA_DIR``) and cartopy's
   download cache, exactly as without nicefigs;
2. the folder in ``$NICEFIGS_NATURAL_EARTH_DIR``, laid out like cartopy's data dir
   (``<dir>/shapefiles/natural_earth/physical/ne_10m_land.shp``, ...);
3. the data bundled with nicefigs (110m and 50m);
4. a download.

The 10m resolution is not bundled; fetch it once with :func:`cache_natural_earth` or
``python -m nicefigs.naturalearth``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ENV_VAR = "NICEFIGS_NATURAL_EARTH_DIR"
"""Environment variable naming a Natural Earth folder searched before the bundled data
and used as the default target of :func:`cache_natural_earth`."""

#: ``(category, name)`` of every Natural Earth layer the map functions draw.
LAYERS: tuple[tuple[str, str], ...] = (
    ("physical", "coastline"),
    ("physical", "land"),
    ("physical", "ocean"),
    ("cultural", "admin_0_boundary_lines_land"),
)
RESOLUTIONS: tuple[str, ...] = ("110m", "50m", "10m")

_BUNDLED = Path(__file__).parent / "_data"


def _search_dirs() -> list[Path]:
    """Folders searched for a missing shapefile, in order."""
    dirs = []
    if os.environ.get(ENV_VAR):
        dirs.append(Path(os.environ[ENV_VAR]).expanduser() / "shapefiles")
    dirs.append(_BUNDLED)
    return dirs


def use_offline_natural_earth() -> None:
    """Make cartopy fall back to local Natural Earth data before it downloads.

    Idempotent and called by the map functions. Files cartopy already finds through its
    own configuration are never replaced.
    """
    from cartopy import config

    downloader = config["downloaders"].get(("shapefiles", "natural_earth"))
    if downloader is None or getattr(downloader, "_nicefigs_offline", False):
        return
    download = downloader.acquire_resource

    def acquire_resource(target_path: Any, format_dict: dict[str, Any]) -> Any:
        rel = Path("natural_earth") / format_dict["category"]
        rel = rel / f"ne_{format_dict['resolution']}_{format_dict['name']}.shp"
        for root in _search_dirs():
            candidate = root / rel
            if candidate.exists():
                return candidate
        return download(target_path, format_dict)

    downloader.acquire_resource = acquire_resource
    downloader._nicefigs_offline = True


def cache_natural_earth(
    path: str | os.PathLike[str] | None = None, resolutions: Sequence[str] = RESOLUTIONS
) -> Path:
    """Download every Natural Earth layer nicefigs may draw, so maps work offline later.

    Run once on a machine with internet access. Layers already present are not fetched
    again.

    Args:
        path: Target folder, laid out like cartopy's data dir. Defaults to
            ``$NICEFIGS_NATURAL_EARTH_DIR``, else cartopy's own data dir
            (``cartopy.config["data_dir"]``). Point ``$NICEFIGS_NATURAL_EARTH_DIR`` (or
            ``$CARTOPY_DATA_DIR``) at it on the offline machine.
        resolutions: Subset of ``"110m"``, ``"50m"``, ``"10m"``.

    Returns:
        The folder the data was cached in.
    """
    import cartopy
    from cartopy.io.shapereader import natural_earth

    unknown = set(resolutions) - set(RESOLUTIONS)
    if unknown:
        raise ValueError(f"unknown resolutions {sorted(unknown)}; choose from {RESOLUTIONS}")
    if path is None:
        path = os.environ.get(ENV_VAR) or cartopy.config["data_dir"]
    target = Path(path).expanduser()
    old = cartopy.config["data_dir"]
    cartopy.config["data_dir"] = target
    try:
        for resolution in resolutions:
            for category, name in LAYERS:
                natural_earth(resolution=resolution, category=category, name=name)
    finally:
        cartopy.config["data_dir"] = old
    return target


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry: ``python -m nicefigs.naturalearth [PATH] [-r 50m 10m]``."""
    parser = argparse.ArgumentParser(description=cache_natural_earth.__doc__.split("\n\n")[0])
    parser.add_argument(
        "path", nargs="?", default=None, help=f"target folder (default: ${ENV_VAR} or cartopy's)"
    )
    parser.add_argument("-r", "--resolutions", nargs="+", default=list(RESOLUTIONS), choices=RESOLUTIONS)
    args = parser.parse_args(argv)
    print(cache_natural_earth(args.path, args.resolutions))


if __name__ == "__main__":
    main()
