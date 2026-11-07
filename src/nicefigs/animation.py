"""Animations of map and image sequences, saved as MP4 (ffmpeg) or GIF (pillow).

:func:`animate_maps` builds an in-process :class:`~matplotlib.animation.FuncAnimation`
over a ``(T, H, W)`` stack with one fixed colour scale; save it with
:func:`save_animation`. :func:`render_animation` instead draws each frame with a
``draw(i) -> Figure`` callback, optionally across processes, and stitches the frames
with :func:`stitch_frames`; use it for expensive or multi-panel frames.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .palettes import cmap as resolve_cmap
from .palettes import robust_limits
from .style import get_theme

if TYPE_CHECKING:
    from matplotlib.animation import Animation, FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def animate_maps(
    lon: ArrayLike,
    lat: ArrayLike,
    frames: ArrayLike,
    *,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = True,
    diverging: bool = False,
    projection: str = "robinson",
    antarctica_cutoff: bool = True,
    extent: Sequence[float] | None = None,
    land: bool = True,
    coastline: bool = True,
    titles: Sequence[str] | None = None,
    suptitle: str = "",
    colorbar_label: str = "",
    fps: float = 8,
    width: str | float = "half",
    aspect: float = 0.62,
    as_map: bool = True,
) -> tuple[Figure, Axes, FuncAnimation]:
    """Animate a ``(T, H, W)`` stack of gridded fields with one fixed colour scale.

    Args:
        lon: Longitudes in degrees east, 1-D ``(W,)`` or 2-D ``(H, W)``.
        lat: Latitudes in degrees north, 1-D ``(H,)`` or 2-D ``(H, W)``.
        frames: Field stack of shape ``(T, H, W)``.
        cmap: Colormap name; ``"inferno"`` becomes ``"RdBu_r"`` when ``diverging``.
        vmin: Lower colour limit; computed over all frames if ``None``.
        vmax: Upper colour limit; computed over all frames if ``None``.
        robust: Use the 2nd/98th percentiles rather than min/max for missing limits.
        diverging: Use limits symmetric about zero and a diverging colormap.
        projection: Map projection name, as in :func:`nicefigs.make_map_axes`.
        antarctica_cutoff: Crop the southern edge when ``extent`` is ``None``.
        extent: ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
        land: Draw the land background.
        coastline: Draw coastlines.
        titles: Per-frame titles (e.g. dates), one per frame.
        suptitle: Figure title, constant across frames.
        colorbar_label: Colorbar label.
        fps: Playback rate in frames per second.
        width: Figure width, a named width or inches (see :func:`nicefigs.figure_size`).
        aspect: Figure height / width.
        as_map: Draw on projected map axes; ``False`` draws a plain image.

    Returns:
        ``(fig, ax, anim)``; keep a reference to ``anim`` until it is saved.
    """
    from matplotlib.animation import FuncAnimation

    th = get_theme()
    frames = np.asarray(frames, dtype=float)
    if cmap == "inferno" and diverging:
        cmap = "RdBu_r"
    if vmin is None or vmax is None:
        if diverging:
            from .palettes import symmetric_limits

            lo, hi = symmetric_limits(frames)
        elif robust:
            lo, hi = robust_limits(frames)
        else:
            lo, hi = float(np.nanmin(frames)), float(np.nanmax(frames))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    from .figure import figure_size

    fig = plt.figure(figsize=figure_size(width, aspect=aspect), dpi=th.dpi)

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if as_map:
        from .maps import _apply_extent, draw_background, make_map_axes

        ax, ccrs = make_map_axes(fig, 111, projection=projection)
        draw_background(ax, ccrs, land=land, coastline=coastline)
        _apply_extent(ax, ccrs, extent, antarctica_cutoff and extent is None)
        tkw = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
        mesh = ax.pcolormesh(
            lon, lat, frames[0], cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax, shading="nearest", **tkw
        )
        is_mesh = True
    else:
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        mesh = ax.imshow(frames[0], cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax, animated=True)
        is_mesh = False

    from .colorbars import colorbar

    colorbar(mesh, ax, label=colorbar_label)
    ttl = ax.set_title(titles[0] if titles else "", fontsize=th.fs_title, loc="left", pad=3)
    if suptitle:
        fig.suptitle(suptitle, fontsize=th.fs_title)

    def update(i: int) -> tuple:
        if is_mesh:
            mesh.set_array(frames[i].ravel())
        else:
            mesh.set_data(frames[i])
        if titles:
            ttl.set_text(titles[i])
        return (mesh, ttl)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    return fig, ax, anim


def save_animation(anim: Animation, path: str | os.PathLike, *, fps: float = 8, dpi: float = 120) -> Path:
    """Save an animation as GIF (pillow) or, for any other suffix, as video via ffmpeg.

    Args:
        anim: Animation to save, e.g. from :func:`animate_maps`.
        path: Output path; parent directories are created.
        fps: Frames per second.
        dpi: Resolution in dots per inch.

    Returns:
        The path written; a ``.gif`` next to ``path`` if ffmpeg is unavailable or fails.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".gif":
        from matplotlib.animation import PillowWriter

        anim.save(str(p), writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        try:
            from matplotlib.animation import FFMpegWriter

            anim.save(str(p), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            gif = p.with_suffix(".gif")
            from matplotlib.animation import PillowWriter

            anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=dpi)
            return gif
    return p


def stitch_frames(
    frame_dir: str | os.PathLike,
    path: str | os.PathLike,
    *,
    fps: float = 8,
    pattern: str = "frame_%06d.png",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Stitch a directory of PNG frames into an MP4 (ffmpeg) or GIF (pillow).

    An ``.mp4`` target is encoded with ffmpeg (libx264, yuv420p, padded to even
    dimensions); any other suffix, or a missing or failing ffmpeg, produces a GIF.

    Args:
        frame_dir: Directory holding frames named per ``pattern``.
        path: Output path; parent directories are created.
        fps: Frames per second.
        pattern: ``printf``-style frame filename with one integer field.
        crf: ffmpeg constant rate factor; lower is higher quality.
        preset: ffmpeg x264 speed preset.

    Returns:
        The path written; its suffix is ``.gif`` on the GIF fallback.

    Raises:
        FileNotFoundError: No frames match ``pattern`` (GIF path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(str(frame_dir), pattern),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return path
        except (FileNotFoundError, subprocess.CalledProcessError):
            path = path.with_suffix(".gif")
    return _stitch_gif(frame_dir, path, fps=fps, pattern=pattern)


def _stitch_gif(frame_dir: str | os.PathLike, path: Path, *, fps: float, pattern: str) -> Path:
    import glob

    from PIL import Image

    path = Path(path)
    glob_pat = pattern
    if "%" in pattern:  # frame_%06d.png -> frame_*.png
        pre, _, post = pattern.partition("%")
        glob_pat = pre + "*" + post[post.index("d") + 1 :] if "d" in post else pre + "*"
    files = sorted(glob.glob(os.path.join(str(frame_dir), glob_pat)))
    if not files:
        raise FileNotFoundError(f"no frames matching {glob_pat!r} in {frame_dir}")
    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=max(int(1000 / fps), 1), loop=0)
    return path


def _render_one(draw: Callable[[int], Figure], i: int, out_path: str, dpi: float, headless: bool) -> str:
    """Draw frame ``i`` and save it to ``out_path``; module-level so it pickles to workers."""
    import gc

    if headless:
        import matplotlib

        matplotlib.use("Agg")
    fig = draw(i)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    del fig
    gc.collect()
    return out_path


def render_animation(
    draw: Callable[[int], Figure],
    n_frames: int,
    path: str | os.PathLike,
    *,
    fps: float = 8,
    dpi: float = 120,
    max_workers: int | None = None,
    method: Literal["parallel", "sequential"] = "parallel",
    pattern: str = "frame_%06d.png",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Render frames with a ``draw(i) -> Figure`` callback and stitch them into a video.

    Each returned figure is saved and closed, then all frames are stitched with
    :func:`stitch_frames`. Unlike :func:`animate_maps`, a frame may contain anything.

    Args:
        draw: Callable returning the figure for frame ``i``. With
            ``method="parallel"`` it must be picklable: a module-level function or
            a :func:`functools.partial` of one, not a lambda or closure.
        n_frames: Number of frames.
        path: Output path (``.mp4`` preferred, GIF fallback).
        fps: Frames per second.
        dpi: Frame resolution in dots per inch.
        max_workers: Process cap for ``"parallel"``; default ``min(n_frames, cpus, 8)``.
        method: ``"parallel"`` renders in a process pool, ``"sequential"`` in-process.
        pattern: ``printf``-style frame filename, as in :func:`stitch_frames`.
        crf: ffmpeg constant rate factor, as in :func:`stitch_frames`.
        preset: ffmpeg x264 preset, as in :func:`stitch_frames`.

    Returns:
        The path written, as returned by :func:`stitch_frames`.
    """
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nicefigs_anim_") as td:
        if method == "parallel" and n_frames > 1:
            from concurrent.futures import ProcessPoolExecutor

            mw = max_workers or min(n_frames, (os.cpu_count() or 4), 8)
            with ProcessPoolExecutor(max_workers=mw) as ex:
                futures = [
                    ex.submit(_render_one, draw, i, os.path.join(td, pattern % i), dpi, True)
                    for i in range(n_frames)
                ]
                for f in futures:
                    f.result()
        else:
            for i in range(n_frames):
                _render_one(draw, i, os.path.join(td, pattern % i), dpi, False)
        return stitch_frames(td, path, fps=fps, pattern=pattern, crf=crf, preset=preset)
