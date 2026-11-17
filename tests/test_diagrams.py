"""Contract of `flux_diagram` and `flux_legend`: every flux an arrow pointing its way, sized by √|value|."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import data


@pytest.fixture
def budget():
    return data.box_budget()


def _draw(b, **kw):
    return nf.flux_diagram(
        b["storage"],
        lateral=b["lateral"],
        vertical=b["vertical"],
        ground=b["ground"],
        parts=b["parts"],
        row_labels=b["row_labels"],
        col_labels=b["col_labels"],
        **kw,
    )


def test_the_example_budget_closes(budget):
    """Storage change = lateral inflow + flux in through the bottom − flux out through the top."""
    up_in = np.vstack([budget["ground"][None, :], budget["vertical"]])
    up_out = np.vstack([budget["vertical"], np.zeros((1, budget["storage"].shape[1]))])
    np.testing.assert_allclose(budget["storage"], budget["lateral"] + up_in - up_out, atol=1e-12)


def test_every_finite_flux_is_one_arrow(budget):
    n_rows, n_cols = budget["storage"].shape
    vertical = budget["vertical"].copy()
    vertical[0, 1] = np.nan
    budget["parts"] = {}
    _, d = _draw({**budget, "vertical": vertical})
    assert len(d.of_kind("vertical")) == (n_rows - 1) * n_cols - 1, "the NaN flux is not drawn"
    assert len(d.of_kind("lateral")) == 2 * n_rows * n_cols, "both sides of every box"
    assert len(d.of_kind("ground")) == n_cols
    assert d.boxes.shape == (n_rows, n_cols)


def test_arrows_point_the_way_the_flux_goes(budget):
    _, d = _draw(budget)
    for a in d.of_kind("vertical") + d.of_kind("ground"):
        assert np.sign(a.head[1] - a.tail[1]) == np.sign(a.value) or a.value == 0
        assert a.head[0] == a.tail[0]
    for a in d.of_kind("lateral"):
        k, j = a.index
        centre = j + 0.5
        inward = abs(a.head[0] - centre) < abs(a.tail[0] - centre)
        assert inward == (a.value >= 0), "in when converging, out when diverging"
        assert a.head[1] == a.tail[1] == k


def test_horizontal_fluxes_point_right_when_positive():
    _, d = nf.flux_diagram(np.zeros((2, 3)), horizontal=[[1.0, -2.0], [0.5, -0.5]])
    for a in d.of_kind("horizontal"):
        assert np.sign(a.head[0] - a.tail[0]) == np.sign(a.value)
        k, j = a.index
        assert j + 0.5 < min(a.head[0], a.tail[0]) and max(a.head[0], a.tail[0]) < j + 1.5, (
            "between the boxes"
        )


def test_the_drawn_tip_is_at_the_head():
    """The patch, as rendered, ends at the declared head: its farthest vertex along the flux."""
    ax, d = nf.flux_diagram(np.zeros((2, 2)), vertical=[[3.0, -3.0]])
    ax.figure.canvas.draw()
    for a in d.of_kind("vertical"):
        verts = a.artist.get_path().vertices  # data coordinates
        along = np.sign(a.value) * verts[:, 1]
        np.testing.assert_allclose(verts[np.argmax(along)], a.head, atol=1e-3)


def test_widths_grow_as_the_square_root_of_the_value():
    values = np.array([[0.0, 1.0, 4.0, 9.0, 16.0]])
    _, d = nf.flux_diagram(np.zeros((2, 5)), vertical=values, max_width=5.0)
    w = np.array([a.width for a in d.of_kind("vertical")])
    assert w[-1] == pytest.approx(5.0), "the largest value is drawn max_width wide"
    np.testing.assert_allclose(np.diff(w[1:]), np.diff(w[1:])[0], err_msg="equal steps in √value")
    assert w[0] > 0, "a zero flux is still visible"


def test_a_shared_reference_gives_shared_widths():
    _, small = nf.flux_diagram(np.zeros((1, 2)), ground=[1.0, 2.0], reference={"ground": 8.0})
    _, large = nf.flux_diagram(np.zeros((1, 2)), ground=[1.0, 8.0], reference={"ground": 8.0})
    assert small.of_kind("ground")[0].width == large.of_kind("ground")[0].width


def test_box_colours_are_centred_on_zero():
    from matplotlib.colors import to_rgba

    _, d = nf.flux_diagram([[0.0, -2.0, 1.0]], cmap="BrBG_r")
    cm = plt.get_cmap("BrBG_r")
    fills = [to_rgba(b.get_facecolor()) for b in d.boxes.ravel()]
    np.testing.assert_allclose(fills[0], cm(0.5), atol=1e-6)
    np.testing.assert_allclose(fills[1], cm(0.0), atol=1e-6)
    np.testing.assert_allclose(fills[2], cm(0.75), atol=1e-6)


def test_a_given_norm_is_used():
    norm = nf.diverging_norm(-10.0, 10.0)
    _, d = nf.flux_diagram([[1.0]], norm=norm)
    assert d.mappable.norm is norm


def test_the_layout_has_no_overlapping_text(budget):
    ax, _ = _draw(budget, col_title="Region")
    assert nf.check_text_overlaps(ax.figure, clipped=False) == []


@pytest.mark.parametrize(
    "kw",
    [
        {"vertical": np.zeros((3, 2))},
        {"ground": np.zeros(3)},
        {"lateral": np.zeros((2, 2)), "horizontal": np.zeros((2, 1))},
        {"parts": {"c": np.zeros((1, 2))}},
        {"row_labels": ["only one"]},
    ],
)
def test_inconsistent_inputs_are_rejected(kw):
    with pytest.raises(ValueError):
        nf.flux_diagram(np.zeros((2, 2)), **kw)


def test_the_legend_lists_what_it_is_given(budget):
    _, d = _draw(budget)
    labels = {"box": "storage", "vertical": "up", "ground": "ground"}
    legend = nf.flux_legend(d, labels, parts={"c": "convective"})
    assert [t.get_text() for t in legend.get_texts()] == ["storage", "up", "ground", "convective"]
    with pytest.raises(KeyError):
        nf.flux_legend(d, {"sideways": "?"})


def test_an_arrow_is_as_wide_in_points_on_any_aspect():
    """The shaft width is physical: the same on a squashed and a stretched axes."""
    widths = []
    for aspect in (0.3, 3.0):
        _, ax = nf.subplots(width=3.0, aspect=aspect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        arrow = nf.diagram_arrow(ax, (0.1, 0.5), (0.9, 0.5), width=4.0, head_width=4.0)
        ax.figure.canvas.draw()
        y = ax.transData.transform(arrow.get_path().vertices)[:, 1]
        widths.append((y.max() - y.min()) * 72.0 / ax.figure.dpi)
    np.testing.assert_allclose(widths, 4.0, rtol=0.02)


def test_an_arrow_label_sits_at_the_midpoint():
    _, ax = nf.subplots()
    nf.diagram_arrow(ax, (0.0, 0.0), (2.0, 4.0), label="flux")
    (label,) = ax.texts
    assert label.xy == (1.0, 2.0) and label.get_text() == "flux"


def test_box_text_reads_on_its_fill():
    from matplotlib.colors import to_rgba

    _, ax = nf.subplots()
    nf.diagram_box(ax, (0.5, 0.5), 0.4, 0.2, text="dark", facecolor="#003040")
    nf.diagram_box(ax, (0.5, 0.2), 0.4, 0.2, text="light", facecolor="#f0f0f0")
    dark, light = ax.texts
    assert to_rgba(dark.get_color()) == to_rgba("white")
    assert sum(to_rgba(light.get_color())[:3]) < 0.5
