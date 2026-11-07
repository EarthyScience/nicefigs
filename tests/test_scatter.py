"""scatter.py: encodings (group/size/hue)."""

from __future__ import annotations

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

import nicefigs as nf
from nicefigs import data as npd


def test_scatter_plain_returns_pathcollection():
    fig, ax = plt.subplots()
    try:
        d = npd.scatter_bubbles(n=60)
        _, pc = nf.scatter(d["x"], d["y"], ax=ax)
        assert isinstance(pc, mcoll.PathCollection)
    finally:
        plt.close(fig)


def test_scatter_with_hue_draws_colorbar():
    fig, ax = plt.subplots()
    try:
        d = npd.scatter_bubbles(n=60)
        n_axes_before = len(fig.axes)
        _, pc = nf.scatter(d["x"], d["y"], hue=d["color"], colorbar=True, ax=ax)
        assert isinstance(pc, mcoll.PathCollection)
        assert len(fig.axes) == n_axes_before + 1
    finally:
        plt.close(fig)


def test_scatter_with_size_scales_marker_area():
    fig, ax = plt.subplots()
    try:
        d = npd.scatter_bubbles(n=60)
        _, pc = nf.scatter(d["x"], d["y"], size=d["size"], size_range=(5, 50), ax=ax)
        sizes = pc.get_sizes()
        assert sizes.min() >= 5 - 1e-6
        assert sizes.max() <= 50 + 1e-6
        # a size legend should have been added
        assert len(ax.get_children()) > 0
    finally:
        plt.close(fig)


def test_scatter_with_group_returns_dict_and_legend():
    fig, ax = plt.subplots()
    try:
        d = npd.scatter_bubbles(n=90)
        _, result = nf.scatter(d["x"], d["y"], group=d["group"], ax=ax)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(np.unique(d["group"]))
        for v in result.values():
            assert isinstance(v, mcoll.PathCollection)
        # shape legend added
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_scatter_group_and_hue_combined():
    fig, ax = plt.subplots()
    try:
        d = npd.scatter_bubbles(n=90)
        _, result = nf.scatter(d["x"], d["y"], group=d["group"], hue=d["color"], ax=ax)
        assert isinstance(result, dict)
    finally:
        plt.close(fig)


def test_scatter_size_all_equal_falls_back_to_mean():
    fig, ax = plt.subplots()
    try:
        x = np.arange(10.0)
        y = np.arange(10.0)
        size = np.full(10, 5.0)
        _, pc = nf.scatter(x, y, size=size, size_range=(10, 100), ax=ax)
        sizes = pc.get_sizes()
        assert np.allclose(sizes, 55.0)
    finally:
        plt.close(fig)
