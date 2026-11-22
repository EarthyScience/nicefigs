"""Contract of the concept-figure primitives: inch canvas, theme colours, drawn objects and checked save."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs.artists import NICE_COLORS


def test_canvas_data_units_are_inches():
    fig, ax = nf.schematic_axes(3.0, 1.5)
    assert tuple(fig.get_size_inches()) == (3.0, 1.5)
    assert ax.get_xlim() == (0.0, 3.0)
    assert ax.get_ylim() == (0.0, 1.5)
    # one data unit is one inch on the page
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((1, 1))
    assert (x1 - x0) / fig.dpi == pytest.approx(1.0)
    assert (y1 - y0) / fig.dpi == pytest.approx(1.0)


def test_tint_and_shade_blend_toward_white_and_black():
    assert nf.tint("#336699", 0.0) == pytest.approx((1, 1, 1))
    assert nf.tint("#336699", 1.0) == pytest.approx(plt.matplotlib.colors.to_rgb("#336699"))
    assert nf.shade("#336699", 1.0) == pytest.approx((0, 0, 0))
    assert nf.shade("#336699", 0.0) == pytest.approx(plt.matplotlib.colors.to_rgb("#336699"))
    lighter = np.asarray(nf.tint("#336699", 0.3))
    assert (lighter > np.asarray(plt.matplotlib.colors.to_rgb("#336699"))).all()


def test_numbered_disc_defaults_to_the_house_orange_and_shows_its_number():
    fig, ax = nf.schematic_axes(1.0, 1.0)
    disc = nf.numbered_disc(ax, (0.5, 0.5), 3)
    assert disc.get_facecolor()[:3] == pytest.approx(plt.matplotlib.colors.to_rgb(NICE_COLORS["orange"]))
    assert [t.get_text() for t in ax.texts] == ["3"]


def test_group_band_sits_below_what_it_groups_and_carries_its_title():
    fig, ax = nf.schematic_axes(2.0, 1.0)
    box = nf.diagram_box(ax, (1.0, 0.5), 0.5, 0.3, text="x")
    band = nf.group_band(ax, (0.2, 0.1), 1.6, 0.8, "streamed")
    assert band.get_zorder() < box.get_zorder()
    assert "streamed" in [t.get_text() for t in ax.texts]
    assert nf.check_text_overlaps(fig) == []


def test_label_pair_stacks_name_over_detail():
    fig, ax = nf.schematic_axes(2.0, 1.0)
    name, detail = nf.label_pair(ax, (1.0, 0.5), "read", "from zarr")
    assert name.get_position()[1] > detail.get_position()[1]
    assert detail.get_fontsize() < name.get_fontsize()


def test_grid_cells_tile_the_requested_rectangle_and_colour_single_cells():
    fig, ax = nf.schematic_axes(2.0, 2.0)
    cells = nf.grid_cells(ax, (0.5, 0.25), 4, 3, 0.2, fills={(1, 2): "red"})
    assert len(cells) == 12
    xs = sorted({round(c.get_x(), 6) for c in cells})
    ys = sorted({round(c.get_y(), 6) for c in cells})
    assert xs == pytest.approx([0.5, 0.7, 0.9, 1.1])
    assert ys == pytest.approx([0.25, 0.45, 0.65])
    red = [c for c in cells if c.get_facecolor()[:3] == (1.0, 0.0, 0.0)]
    assert len(red) == 1
    assert (red[0].get_x(), red[0].get_y()) == pytest.approx((0.7, 0.65))


def test_folder_icon_stays_inside_its_box():
    fig, ax = nf.schematic_axes(1.0, 1.0)
    icon = nf.folder_icon(ax, (0.2, 0.3), 0.4, 0.3)
    pts = np.asarray(icon.get_xy())
    assert pts[:, 0].min() == pytest.approx(0.2)
    assert pts[:, 0].max() == pytest.approx(0.6)
    assert pts[:, 1].min() == pytest.approx(0.3)
    assert pts[:, 1].max() == pytest.approx(0.6)


def test_save_with_check_text_refuses_overlapping_text(tmp_path):
    fig, ax = nf.schematic_axes(2.0, 1.0)
    ax.text(1.0, 0.5, "first label")
    ax.text(1.02, 0.5, "second label")
    with pytest.raises(ValueError, match="text layout"):
        nf.save(fig, tmp_path / "bad", formats=("png",), check_text=True)
    assert not (tmp_path / "bad.png").exists()


def test_save_with_check_text_writes_a_clean_figure(tmp_path):
    fig, ax = nf.schematic_axes(2.0, 1.0)
    nf.diagram_box(ax, (1.0, 0.5), 1.0, 0.4, text="ok")
    paths = nf.save(fig, tmp_path / "good", formats=("svg", "png"), check_text=True)
    assert [p.suffix for p in paths] == [".svg", ".png"]
    assert all(p.stat().st_size > 0 for p in paths)
    # SVG text is converted to paths, so no <text> element refers to an installed font
    assert "<text" not in paths[0].read_text()
