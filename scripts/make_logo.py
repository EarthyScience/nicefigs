"""Generate the nicefigs logo: a kernel density plot whose iso-density bands form a halved fig.

The idea
--------
The mark is a 2D density estimate drawn the way nicefigs draws one: smooth filled ``contourf`` bands with thin
``contour`` lines in a slightly darker tone of each band. Read as a fig, the bands are the halved fruit:

* the outermost density level *is* the silhouette: one smooth closed spline (broad rounded body, concave shoulder,
  short rounded neck tip towards the upper right, tilted as in a photo of a halved fig). No stem, no outline;
* a thin green skin, then a broad pale cream rind (thicker towards the neck),
* then peach-pink flesh, coral and red around a slim almond cavity along the long axis, offset towards the broad
  lower-left end, with a small deep-purple core.

The "data" is a designed smooth field: the exact distance to the silhouette spline (so rind levels follow it
cleanly) plus a chain of anisotropic Gaussians (the cavity) and a faint organic wobble that only touches the inner
levels. Colours come from one continuous ``fig`` colormap built from the house palettes.

Run from the repository root with ``python scripts/make_logo.py [OUT_DIR]`` (default ``docs/assets/logo``), or
``make logo``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath
from scipy import interpolate
from scipy.spatial import cKDTree

import nicefigs as nf

ANGLE = -50.0  # degrees; rotates the upright fig (neck up) so the neck points to the upper right
PURPLE = "#924099"  # house purple (klimt[5])
BACKING = "#fffdf5"  # pale disc that makes the mark readable on a teal header

# --------------------------------------------------------------------------------------------------
# The silhouette: one smooth closed spline in the upright frame (x across, y along the axis, base at y=0)
# --------------------------------------------------------------------------------------------------

#: Right half of the outline from the base to the neck tip (x, y); mirrored for the left half.
OUTLINE = (
    (0.00, 0.000),
    (0.17, 0.010),
    (0.31, 0.090),
    (0.375, 0.250),
    (0.370, 0.420),
    (0.310, 0.560),
    (0.225, 0.690),
    (0.140, 0.790),
    (0.098, 0.865),
    (0.082, 0.930),
    (0.00, 0.975),
)


def silhouette(n: int = 2000) -> np.ndarray:
    """Dense ``(n, 2)`` points of the smooth closed outline (periodic cubic spline through ``OUTLINE``)."""
    right = np.array(OUTLINE)
    left = right[-2:0:-1] * np.array([-1.0, 1.0])
    pts = np.vstack([right, left, right[:1]])
    tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    x, y = interpolate.splev(np.linspace(0, 1, n), tck)
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------------------------------
# The density field
# --------------------------------------------------------------------------------------------------

#: Anisotropic Gaussians forming the seed cavity: (x, y, sd_x, sd_y, weight). A slim almond along the axis, widest
#: towards the base (lower-left in the tilted view) and pointed towards the neck.
CAVITY = (
    (-0.010, 0.250, 0.108, 0.130, 1.00),
    (0.005, 0.380, 0.118, 0.165, 1.00),
    (0.022, 0.520, 0.098, 0.165, 0.84),
    (0.038, 0.640, 0.070, 0.140, 0.50),
    (0.048, 0.750, 0.040, 0.095, 0.22),
)
RIND_PLATEAU = 0.26  # field value reached across the rind; the flesh levels start just above
RIND_WIDTH = 0.088  # distance from the silhouette over which the rind field rises


@dataclass(frozen=True)
class Design:
    """One logo candidate: iso-density levels (fractions of the field maximum), first = silhouette."""

    name: str
    levels: tuple[float, ...]
    rind_n: int = 4  # how many of the lowest levels are rind bands (skin, green-cream, cream, pale cream)
    lines: bool = True  # thin darker contour line per level
    line_w: float = 0.55  # contour line width in points at a 4 inch mark
    wobble: float = 0.03  # organic wobble of the inner levels
    seed: int = 4


def _wobble_field(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = rng.normal(0, 8.0, size=(7, 2))
    ph = rng.uniform(0, 2 * np.pi, 7)
    return sum(np.sin(k[i, 0] * x + k[i, 1] * y + ph[i]) for i in range(7)) / 3.0


def density(d: Design, n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the field on an upright grid; returns ``X, Y`` (rotated to the tilted view) and ``F`` in [0, 1]."""
    xs = np.linspace(-0.46, 0.46, n)
    ys = np.linspace(-0.09, 1.10, int(n * 1.3))
    X, Y = np.meshgrid(xs, ys)
    outline = silhouette()
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dist, _ = cKDTree(outline).query(pts)
    inside = MPath(outline).contains_points(pts)
    sd = np.where(inside, dist, -dist).reshape(X.shape)  # exact signed distance to the spline
    u = np.clip(
        sd / RIND_WIDTH, -0.2, 1
    )  # signed and continuous across the silhouette, so its level is smooth
    rind = RIND_PLATEAU * np.sign(u) * np.abs(u) ** 0.9
    flesh = sum(
        w * np.exp(-0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2)) for cx, cy, sx, sy, w in CAVITY
    )
    flesh = flesh / flesh.max() * np.clip(sd / 0.03, 0, 1)
    wob = d.wobble * np.clip(flesh / 0.15, 0, 1) * _wobble_field(X, Y, d.seed)  # inner levels only
    f = rind + (1 - RIND_PLATEAU) * flesh + wob
    a = np.deg2rad(ANGLE)
    x0, y0 = 0.0, 0.40
    XR = x0 + (X - x0) * np.cos(a) - (Y - y0) * np.sin(a)
    YR = y0 + (X - x0) * np.sin(a) + (Y - y0) * np.cos(a)
    return XR, YR, f / f.max()


# --------------------------------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------------------------------


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = np.array(mcolors.to_rgb(a)), np.array(mcolors.to_rgb(b))
    return mcolors.to_hex(ca + (cb - ca) * t)


def fig_cmap() -> mcolors.LinearSegmentedColormap:
    """Flesh colormap from the house palettes (hiroshige peach and coral, klimt magenta and purple)."""
    mb = nf.artists.METBREWER
    cream = _mix(mb["hokusai3"][0], "#fffbe8", 0.5)
    peach = _mix(mb["hiroshige"][3], mb["hiroshige"][0], 0.30)
    mid = _mix(mb["klimt"][1], PURPLE, 0.5)
    stops = [
        (0.00, cream),
        (0.14, peach),
        (0.34, mb["hiroshige"][0]),
        (0.52, mb["klimt"][1]),
        (0.68, mid),
        (0.84, PURPLE),
        (1.00, _mix(PURPLE, "#1a0b2e", 0.5)),
    ]
    return mcolors.LinearSegmentedColormap.from_list("fig", stops)


def rind_colours(n: int) -> list[str]:
    """Skin green, then pale green-cream, cream and near-white cream (``n`` = 4), or skin + cream (``n`` = 2)."""
    mb = nf.artists.METBREWER
    green = mb["hokusai3"][1]
    cream = _mix(mb["hokusai3"][0], "#fffbe8", 0.72)
    skin = _mix(_mix(green, "#4f8f3f", 0.2), cream, 0.08)
    if n == 2:
        return [skin, cream]
    return [skin, _mix(green, cream, 0.62), cream, _mix(cream, "#ffffff", 0.35)]


def band_colours(d: Design) -> list[str]:
    """One colour per level: the rind list, then the fig colormap sampled across the flesh levels."""
    cm = fig_cmap()
    nflesh = len(d.levels) - d.rind_n
    return rind_colours(d.rind_n) + [mcolors.to_hex(cm(p)) for p in np.linspace(0.07, 1.0, nflesh)]


# --------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------


def view_window(
    X: np.ndarray, Y: np.ndarray, F: np.ndarray, lv0: float, pad: float = 0.07
) -> tuple[float, ...]:
    """Square window around the silhouette (data units) with ``pad`` of whitespace."""
    m = F > lv0
    lo = np.array([X[m].min(), Y[m].min()])
    hi = np.array([X[m].max(), Y[m].max()])
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    return c[0], c[1], half


def draw_mark(ax, d: Design, size_in: float, backing: str | None = None) -> None:
    """Draw the density plot on ``ax`` (axes off): stacked ``contourf`` bands plus thin ``contour`` lines."""
    X, Y, F = density(d)
    cx, cy, half = view_window(X, Y, F, d.levels[0])
    if backing:  # pale disc around the fig, with room to breathe
        ax.add_patch(mpl.patches.Circle((cx, cy), 1.12 * half, fc=backing, ec="none", zorder=0))
        half *= 1.17
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    cols = band_colours(d)
    for lv, col in zip(d.levels, cols, strict=True):  # stacked, so neighbouring bands never leave seams
        ax.contourf(X, Y, F, levels=[lv, 10], colors=[col], zorder=2)
    if d.lines:
        lc = [_mix(col, "#2a1038", 0.30) for col in cols]
        for i in range(1, len(d.levels)):  # finer lines inside the pale rind, regular lines in the flesh
            w = d.line_w * (0.45 if i < d.rind_n else 1.0) * size_in / 4
            ax.contour(X, Y, F, levels=[d.levels[i]], colors=[lc[i]], linewidths=w, zorder=3)


def render_mark(d: Design, size_in: float = 4.0, backing: str | None = None):
    """Square transparent figure with the mark."""
    fig = plt.figure(figsize=(size_in, size_in))
    fig.patch.set_alpha(0)
    draw_mark(fig.add_axes((0, 0, 1, 1)), d, size_in, backing)
    return fig


# --------------------------------------------------------------------------------------------------
# Candidates, lockups and export
# --------------------------------------------------------------------------------------------------

CANDIDATES = {
    "kde-final": Design("kde-final", (0.02, 0.075, 0.125, 0.19, 0.31, 0.39, 0.48, 0.58, 0.69, 0.80, 0.91)),
}
SMALL = Design(
    "small", (0.02, 0.085, 0.31, 0.45, 0.60, 0.76, 0.91), rind_n=2, lines=False, wobble=0.02
)  # 32 px and below
RECOMMENDED = "kde-final"
THEMES = {
    "light": {"bg": "#ffffff", "nice": _mix(PURPLE, "#000000", 0.35), "figs": "#2c2d54"},
    "dark": {"bg": "#0d1117", "nice": "#df9ed4", "figs": "#e7e5f0"},
}


def render_lockup(d: Design, theme: str, height_in: float = 1.0):
    """Mark followed by the wordmark ``nice`` (bold, house purple) + ``figs``."""
    th = THEMES[theme]
    fs = height_in * 72 * 0.62
    probe = plt.figure(figsize=(4, 1))
    rend = probe.canvas.get_renderer()
    widths = [
        probe.text(0, 0, t, fontsize=fs, fontweight=w).get_window_extent(rend).width / probe.dpi
        for t, w in (("nice", "bold"), ("figs", "normal"))
    ]
    plt.close(probe)
    gap = 0.12 * height_in
    width_in = height_in + gap + sum(widths) + 0.12 * height_in
    fig = plt.figure(figsize=(width_in, height_in))
    fig.patch.set_alpha(0)
    draw_mark(fig.add_axes((0, 0, height_in / width_in, 1)), d, height_in)
    x = (height_in + gap) / width_in
    fig.text(x, 0.30, "nice", fontsize=fs, fontweight="bold", color=th["nice"], ha="left", va="baseline")
    fig.text(x + widths[0] / width_in, 0.30, "figs", fontsize=fs, color=th["figs"], ha="left", va="baseline")
    return fig


def _png(fig, path: Path, px: int) -> None:
    fig.savefig(path, dpi=px / fig.get_figwidth(), transparent=True)
    plt.close(fig)


def write_release(out_dir: Path, name: str = RECOMMENDED) -> None:
    """Write the release files for the recommended candidate (+ small-size lockups and the favicon)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    d = CANDIDATES[name]
    for theme in ("light", "dark"):
        for suffix, dd in (("", d), ("-small", SMALL)):
            fig = render_lockup(dd, theme)
            fig.savefig(out_dir / f"nicefigs-lockup-{theme}{suffix}.svg", transparent=True)
            plt.close(fig)
    fig = render_mark(d, 4.0, BACKING)
    fig.savefig(out_dir / "nicefigs-mark.svg", transparent=True)
    plt.close(fig)
    _png(render_mark(SMALL, 4.0), out_dir / "favicon.png", 64)
    tmp = out_dir / "_fav.png"
    _png(render_mark(SMALL, 4.0), tmp, 512)
    Image.open(tmp).convert("RGBA").save(
        out_dir / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)]
    )
    tmp.unlink()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "out_dir", nargs="?", default="docs/assets/logo", type=Path, help="release output directory"
    )
    args = ap.parse_args(argv)
    with nf.style():
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
        mpl.rcParams["svg.fonttype"] = "path"
        write_release(args.out_dir)


if __name__ == "__main__":
    main()
