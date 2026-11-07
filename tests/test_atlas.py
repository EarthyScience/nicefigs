"""atlas.py: the component panel, and images.overlay_grid that it uses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs.atlas import component_panel


def test_overlay_grid_covers_exactly_its_window_and_hides_masked_cells():
    fig, ax = plt.subplots()
    try:
        nf.image(np.zeros((32, 32, 3)), rgb=True, ax=ax)
        values = np.arange(16.0).reshape(4, 4)
        im = nf.overlay_grid(ax, values, window=(8, 4, 16), mask_below=2.0)
        assert im.get_extent() == pytest.approx([3.5, 19.5, 23.5, 7.5])
        arr = im.get_array()
        assert arr.mask[0, :3].all() and not arr.mask[0, 3]
        assert ax.get_xlim() == pytest.approx((-0.5, 31.5))
    finally:
        plt.close(fig)


def _panel_inputs(n_chips=6, seed=0):
    rng = np.random.default_rng(seed)
    lon, lat = rng.uniform(-170, 170, 400), rng.uniform(-55, 70, 400)
    chips = rng.uniform(0, 0.3, (n_chips, 48, 48, 3))
    overlays = rng.gamma(2.0, 1.0, (n_chips, 4, 4))
    counts, edges = np.histogram(rng.gamma(2.0, 1.0, 500), bins=20)
    return lon, lat, rng.uniform(0, 1, 400), chips, overlays, counts, edges


def test_component_panel_lays_out_map_histogram_text_and_one_axes_per_chip():
    lon, lat, values, chips, overlays, counts, edges = _panel_inputs()
    fig, axes = component_panel(
        lon,
        lat,
        values,
        chips=chips,
        overlays=overlays,
        window=(16, 16, 16),
        hist_edges=edges,
        hist_counts=counts,
        text=["class sparse", "range 120 km"],
        label="wetlands",
    )
    try:
        assert set(axes) == {"map", "hist", "text", "chips", "extra_rows"} and axes["extra_rows"] == []
        assert len(axes["chips"]) == 6
        assert all(len(a.images) == 2 for a in axes["chips"])
        shown = " ".join(t.get_text() for t in axes["text"].texts)
        assert "range 120 km" in shown and "wetlands" in shown
        assert axes["hist"].has_data()
    finally:
        plt.close(fig)


def test_component_panel_without_chips_leaves_an_empty_label_slot():
    lon, lat, values, *_ = _panel_inputs()
    fig, axes = component_panel(lon, lat, values)
    try:
        assert axes["chips"] == []
        assert "label:" in " ".join(t.get_text() for t in axes["text"].texts)
    finally:
        plt.close(fig)


def test_overlays_need_a_window_and_one_per_chip():
    lon, lat, values, chips, overlays, *_ = _panel_inputs()
    with pytest.raises(ValueError):
        component_panel(lon, lat, values, chips=chips, overlays=overlays)
    with pytest.raises(ValueError):
        component_panel(lon, lat, values, chips=chips, overlays=overlays[:2], window=(16, 16, 16))


def test_extra_chip_rows_are_drawn_below_the_first_with_their_titles():
    lon, lat, values, chips, overlays, counts, edges = _panel_inputs()
    extra = [{"chips": chips[:4], "overlays": overlays[:4], "title": "pure cells, heaviest tails"}]
    fig, axes = component_panel(
        lon, lat, values, chips=chips, overlays=overlays, window=(16, 16, 16), extra_rows=extra
    )
    try:
        assert len(axes["extra_rows"]) == 1 and len(axes["extra_rows"][0]) == 4
        assert all(len(a.images) == 2 for a in axes["extra_rows"][0])
        row_y = [a.get_position().y0 for a in (axes["chips"][0], axes["extra_rows"][0][0])]
        assert row_y[1] < row_y[0]
        assert "heaviest tails" in axes["extra_rows"][0][0].get_title(loc="left")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("n_chips", [0, 6])
def test_component_panel_text_stays_clear_of_other_text_and_inside_the_figure(n_chips):
    lon, lat, values, chips, overlays, counts, edges = _panel_inputs()
    fig, axes = component_panel(
        lon,
        lat,
        values,
        chips=chips if n_chips else None,
        overlays=overlays if n_chips else None,
        window=(16, 16, 16),
        hist_edges=edges,
        hist_counts=counts * 1000,
        text=[
            "class dense  ·  rate 1.000",
            "range 14431 km",
            "best map precipitation of the wettest quarter  R² 0.16",
        ],
        label="dense within-cell contrast strongest in subtropical savanna and agriculture mosaics",
        title="component 16",
        map_label="per-cell map",
    )
    try:
        assert nf.check_text_overlaps(fig) == []
        shown = " ".join(t.get_text() for t in axes["text"].texts).replace("\n", " ")
        assert "agriculture mosaics" in shown
    finally:
        plt.close(fig)
