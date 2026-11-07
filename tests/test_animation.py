"""animation.py: animate_maps returns a FuncAnimation; save_animation writes a file."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nicefigs as nf
from nicefigs import data as npd


def _small_cube():
    return npd.map_timeseries_cube(n_frames=4, shape=(12, 24))


def test_animate_maps_returns_funcanimation_and_figure():
    lon, lat, frames = _small_cube()
    fig, _, anim = nf.animate_maps(lon, lat, frames)
    try:
        assert isinstance(anim, FuncAnimation)
        assert fig is not None
        # exercise a couple of frame updates without erroring
        anim._func(0)
        anim._func(1)
    finally:
        plt.close(fig)


def test_animate_maps_diverging():
    lon, lat, frames = _small_cube()
    fig, _, anim = nf.animate_maps(lon, lat, frames, diverging=True)
    try:
        assert isinstance(anim, FuncAnimation)
    finally:
        plt.close(fig)


def test_animate_maps_non_map_mode():
    lon, lat, frames = _small_cube()
    fig, _, anim = nf.animate_maps(lon, lat, frames, as_map=False)
    try:
        assert isinstance(anim, FuncAnimation)
        anim._func(0)
    finally:
        plt.close(fig)


def test_animate_maps_with_titles_and_suptitle():
    lon, lat, frames = _small_cube()
    titles = [f"t{i}" for i in range(frames.shape[0])]
    fig, _, anim = nf.animate_maps(lon, lat, frames, titles=titles, suptitle="cube")
    try:
        assert isinstance(anim, FuncAnimation)
        anim._func(2)
    finally:
        plt.close(fig)


def test_save_animation_gif(tmp_path):
    lon, lat, frames = _small_cube()
    fig, _, anim = nf.animate_maps(lon, lat, frames)
    try:
        out = nf.save_animation(anim, tmp_path / "anim.gif", fps=4)
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)
