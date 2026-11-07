"""Tests for the frame-based animation engine: stitch_frames + render_animation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf


# Module-level (picklable) frame drawer for the parallel path.
def _demo_draw(i):
    nf.set_theme("paper")
    fig, ax = nf.subplots(width="half", aspect=0.7)
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x + i * 0.4))
    ax.set_title(f"frame {i}")
    return fig


def test_stitch_frames_gif(tmp_path):
    # write a few tiny PNG frames, then stitch to gif
    for i in range(4):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(np.full((8, 8), i))
        fig.savefig(tmp_path / f"frame_{i:06d}.png", dpi=30)
        plt.close(fig)
    out = nf.stitch_frames(tmp_path, tmp_path / "out.gif", fps=4)
    assert out.exists() and out.suffix == ".gif" and out.stat().st_size > 0


def test_stitch_frames_mp4_falls_back_to_gif_without_ffmpeg(tmp_path, monkeypatch):
    for i in range(3):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(np.full((8, 8), i))
        fig.savefig(tmp_path / f"frame_{i:06d}.png", dpi=30)
        plt.close(fig)

    import subprocess

    def _boom(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", _boom)
    out = nf.stitch_frames(tmp_path, tmp_path / "movie.mp4", fps=4)
    assert out.suffix == ".gif" and out.exists()


def test_render_animation_sequential(tmp_path):
    out = nf.render_animation(_demo_draw, 4, tmp_path / "seq.gif", fps=4, method="sequential", dpi=40)
    assert out.exists() and out.stat().st_size > 0


def test_render_animation_parallel(tmp_path):
    out = nf.render_animation(
        _demo_draw, 4, tmp_path / "par.gif", fps=4, method="parallel", max_workers=2, dpi=40
    )
    assert out.exists() and out.stat().st_size > 0
