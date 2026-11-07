"""images.py: image, heatmap, image_grid."""

from __future__ import annotations

import matplotlib.image as mimage
import matplotlib.pyplot as plt

import nicefigs as nf
from nicefigs import data as npd


def test_image_intensity_returns_axesimage():
    fig, ax = plt.subplots()
    try:
        arr = npd.heatmap_matrix(shape=(20, 20))
        _, im = nf.image(arr, robust=True, title="t", ax=ax)
        assert isinstance(im, mimage.AxesImage)
        assert ax.get_xticks().size == 0 or list(ax.get_xticks()) == []
    finally:
        plt.close(fig)


def test_image_rgb_returns_none_mappable():
    fig, ax = plt.subplots()
    try:
        img = npd.rgb_patch(size=32)
        _, m = nf.image(img, rgb=True, stretch=True, ax=ax)
        assert m is None
        assert len(ax.images) == 1
    finally:
        plt.close(fig)


def test_image_auto_detects_rgb_by_shape():
    fig, ax = plt.subplots()
    try:
        img = npd.rgb_patch(size=16)
        _, m = nf.image(img, ax=ax)  # rgb not passed explicitly
        assert m is None
    finally:
        plt.close(fig)


def test_heatmap_smoke_with_labels_and_annotations():
    fig, ax = plt.subplots()
    try:
        mat = npd.heatmap_matrix(shape=(4, 5))
        _, im = nf.heatmap(
            mat,
            row_labels=[f"r{i}" for i in range(4)],
            col_labels=[f"c{j}" for j in range(5)],
            annotate=True,
            title="heat",
            ax=ax,
        )
        assert isinstance(im, mimage.AxesImage)
        assert len(ax.texts) == 4 * 5
    finally:
        plt.close(fig)


def test_heatmap_without_annotate_has_no_text():
    fig, ax = plt.subplots()
    try:
        mat = npd.heatmap_matrix(shape=(3, 3))
        nf.heatmap(mat, annotate=False, ax=ax)
        assert len(ax.texts) == 0
    finally:
        plt.close(fig)


def test_image_grid_single_colorbar():
    imgs = [npd.heatmap_matrix(seed=s, shape=(10, 10)) for s in range(4)]
    fig, axes = nf.image_grid(imgs, colorbar_mode="single", titles=[f"im{i}" for i in range(4)])
    try:
        assert axes.shape[0] * axes.shape[1] >= 4
        # exactly one colorbar axes beyond the panel axes
        n_panel_axes = axes.size
        assert len(fig.axes) == n_panel_axes + 1
    finally:
        plt.close(fig)


def test_image_grid_row_colorbars():
    imgs = [npd.heatmap_matrix(seed=s, shape=(6, 6)) for s in range(4)]
    fig, axes = nf.image_grid(imgs, nrows=2, ncols=2, colorbar_mode="row")
    try:
        n_panel_axes = axes.size
        assert len(fig.axes) == n_panel_axes + 2
    finally:
        plt.close(fig)


def test_image_grid_each_colorbar():
    imgs = [npd.heatmap_matrix(seed=s, shape=(6, 6)) for s in range(3)]
    fig, axes = nf.image_grid(imgs, nrows=1, ncols=3, colorbar_mode="each")
    try:
        n_panel_axes = axes.size
        assert len(fig.axes) == n_panel_axes + 3
    finally:
        plt.close(fig)


def test_image_grid_none_colorbar_mode():
    imgs = [npd.heatmap_matrix(seed=s, shape=(6, 6)) for s in range(2)]
    fig, axes = nf.image_grid(imgs, nrows=1, ncols=2, colorbar_mode="none")
    try:
        assert len(fig.axes) == axes.size
    finally:
        plt.close(fig)


def test_image_grid_rgb_mode():
    imgs = [npd.rgb_patch(seed=s, size=16) for s in range(3)]
    fig, axes = nf.image_grid(imgs, nrows=1, ncols=3, rgb=True)
    try:
        assert axes.shape == (1, 3)
    finally:
        plt.close(fig)


def test_image_grid_handles_extra_empty_cells():
    imgs = [npd.heatmap_matrix(seed=s, shape=(5, 5)) for s in range(3)]
    fig, axes = nf.image_grid(imgs, nrows=2, ncols=2)
    try:
        assert axes.shape == (2, 2)
    finally:
        plt.close(fig)
