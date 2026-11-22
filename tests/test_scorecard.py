"""Contract of the default scorecard: text is the raw value, colour is percent better than a baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf

BAND_COLORS = [
    "#C4372D",
    "#E25C45",
    "#EC8869",
    "#F1B398",
    "#F9DDCE",
    "#F2F2F2",
    "#DEE9F5",
    "#C0D5E9",
    "#94BDD9",
    "#669CC9",
    "#437AB5",
]
EDGES = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]


def _cells(ax):
    """Filled cell patches in reading order."""
    return sorted((p for p in ax.patches if p.get_fill()), key=lambda p: (p.get_xy()[1], p.get_xy()[0]))


def _outlines(ax):
    return [p for p in ax.patches if not p.get_fill()]


def _card(rmse, **kw):
    rmse = np.asarray(rmse, dtype=float)
    ncol = rmse.shape[1]
    cols = [f"c{j}" for j in range(ncol)]
    return nf.scorecard(
        rmse,
        row_labels=["ref"] + [f"m{i}" for i in range(1, rmse.shape[0])],
        col_labels=cols,
        direction=dict.fromkeys(cols, "lower_better"),
        **kw,
    )


# ----------------------------------------------------------------- skill_gap percent
def test_percent_gap_is_relative_to_the_reference_and_positive_when_better():
    # lower_better: 90 against a reference of 100 is 10 % better, 120 is 20 % worse
    gap = nf.skill_gap([[90.0, 120.0]], [[100.0, 100.0]], direction="lower_better", mode="percent")
    assert gap[0] == pytest.approx([10.0, -20.0])
    up = nf.skill_gap([[1.1]], [[1.0]], direction="higher_better", mode="percent")
    assert up[0, 0] == pytest.approx(10.0)


def test_percent_gap_floor_caps_the_denominator_and_zero_reference_is_nan():
    assert nf.skill_gap([[0.05]], [[0.1]], mode="percent", direction="lower_better", floor=1.0)[0, 0] == (
        pytest.approx(5.0)
    )
    assert np.isnan(nf.skill_gap([[1.0]], [[0.0]], mode="percent", direction="lower_better")[0, 0])


def test_percent_gap_of_a_target_metric_is_relative_to_the_reference_distance():
    gap = nf.skill_gap([[1.1]], [[1.4]], mode="percent", direction="target", target=1.0)
    assert gap[0, 0] == pytest.approx(100 * (0.4 - 0.1) / 0.4)


def test_percent_gap_rejects_uncertainty_and_unknown_modes():
    with pytest.raises(ValueError, match="uncertainty"):
        nf.skill_gap([[1.0]], [[1.0]], uncertainty=[[1.0]], mode="percent")
    with pytest.raises(ValueError, match="mode"):
        nf.skill_gap([[1.0]], [[1.0]], mode="relative")


# ------------------------------------------------------------------ the colour scale
@pytest.mark.parametrize(
    "percent, band",
    [
        (-30, 0),
        (-20, 1),
        (-7, 2),
        (-5, 3),
        (-1.5, 4),
        (-0.5, 5),
        (0, 5),
        (1.0, 6),
        (3, 7),
        (7, 8),
        (15, 9),
        (30, 10),
    ],
)
def test_skill_cmap_maps_percent_to_its_band_with_blue_better(percent, band):
    cm, norm = nf.scorecard_skill_cmap()
    assert list(norm.boundaries) == EDGES
    assert cm(norm(percent)) == pytest.approx(mcolors.to_rgba(BAND_COLORS[band]))


def test_skill_cmap_pointed_ends_are_dark_red_worse_and_dark_blue_better():
    cm, norm = nf.scorecard_skill_cmap()
    assert cm(norm(-80)) == pytest.approx(mcolors.to_rgba("#9C2520"))
    assert cm(norm(80)) == pytest.approx(mcolors.to_rgba("#26549A"))


def test_a_cell_is_coloured_by_the_band_of_its_percent_better_than_the_baseline():
    # references 100: values give +30, +7, -0.5, -15, -60 percent (lower is better)
    ax = _card([[100.0] * 5, [70.0, 93.0, 100.5, 115.0, 160.0]], reference_row="ref")
    faces = [p.get_facecolor() for p in _cells(ax)[5:]]
    wanted = ["#437AB5", "#94BDD9", "#F2F2F2", "#E25C45", "#9C2520"]
    for got, want in zip(faces, wanted):
        assert got == pytest.approx(mcolors.to_rgba(want))


def test_better_cells_are_blue_for_both_directions_of_a_metric():
    vals = np.array([[1.0, 1.0], [0.5, 2.0]])
    ax = nf.scorecard(
        vals,
        row_labels=["ref", "m"],
        col_labels=["err", "score"],
        direction={"err": "lower_better", "score": "higher_better"},
        reference_row="ref",
    )
    for patch in _cells(ax)[2:]:  # +50 % and +100 % better: the pointed-end blue
        assert patch.get_facecolor() == pytest.approx(mcolors.to_rgba("#26549A"))


def test_a_named_cmap_of_the_registry_flips_rdbu_so_the_object_must_be_passed():
    # nicefigs.cmap("RdBu") is RdBu_r (red = high): with reference= the better cells come out red.
    assert nf.cmap("RdBu")(0.99)[0] > nf.cmap("RdBu")(0.99)[2]
    vals = np.array([[100.0], [70.0]])
    kw = {
        "row_labels": ["ref", "m"],
        "col_labels": ["e"],
        "direction": {"e": "lower_better"},
        "reference": [[100.0]],
    }
    by_name = _cells(nf.scorecard(vals, cmap="RdBu", **kw))[1].get_facecolor()
    by_object = _cells(nf.scorecard(vals, cmap=mpl.colormaps["RdBu"], **kw))[1].get_facecolor()
    assert by_name[0] > by_name[2]  # red
    assert by_object[2] > by_object[0]  # blue: better


# --------------------------------------------------------------- baseline and states
def test_reference_row_is_one_outline_around_the_row_and_labelled_baseline():
    ax = _card(
        [[1.0, 2.0, 3.0], [0.9, 2.1, 3.0], [1.2, 2.0, 2.0]], reference_row="ref", col_groups=["a", "a", "b"]
    )
    boxes = _outlines(ax)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.get_edgecolor() == pytest.approx(mcolors.to_rgba("#222222"))
    assert box.get_linewidth() == pytest.approx(2.0)
    assert box.get_xy() == (0.0, 0.0)
    assert box.get_width() == pytest.approx(ax.get_xlim()[1])
    assert box.get_height() == pytest.approx(1.0)
    labels = [t.get_text() for t in ax.texts]
    assert "ref (baseline)" in labels
    assert _cells(ax)[0].get_facecolor() == pytest.approx(mcolors.to_rgba("#e8e8e8"))


def test_reference_cells_are_outlined_one_by_one_and_supply_the_reference_per_column():
    vals = np.array([[1.0, 2.0], [0.5, 2.5], [1.5, 1.0]])
    mask = np.array([[True, False], [False, False], [False, True]])
    ax = nf.scorecard(
        vals,
        row_labels=list("abc"),
        col_labels=["x", "y"],
        direction={"x": "lower_better", "y": "lower_better"},
        reference_cells=mask,
    )
    assert len(_outlines(ax)) == 2
    cells = _cells(ax)
    # column x is measured against 1.0, column y against 1.0: 0.5 is +50 % -> pointed-end colour
    assert cells[2].get_facecolor() == pytest.approx(mcolors.to_rgba("#26549A"))
    assert cells[3].get_facecolor() == pytest.approx(mcolors.to_rgba("#9C2520"))
    assert cells[0].get_facecolor() == pytest.approx(mcolors.to_rgba("#e8e8e8"))


def test_reference_cells_need_exactly_one_cell_per_column():
    with pytest.raises(ValueError, match="one cell per column"):
        nf.scorecard(
            np.ones((2, 2)),
            row_labels=["a", "b"],
            col_labels=["x", "y"],
            reference_cells=[[True, False], [False, False]],
        )


def test_missing_values_print_na_on_grey_and_unresolved_print_their_value_in_grey():
    vals = np.array([[1.0, 1.0, 1.0], [0.5, np.nan, 0.995]])
    ax = _card(vals, reference_row="ref", unresolved=[[False] * 3, [False, False, True]])
    cells = _cells(ax)
    texts = [t for t in ax.texts if t.get_text() in ("n/a", "0.995")]
    assert sorted(t.get_text() for t in texts) == ["0.995", "n/a"]
    for t in texts:
        assert t.get_color() == "#8c8c8c"
    assert cells[4].get_facecolor() == pytest.approx(mcolors.to_rgba("#e8e8e8"))
    assert cells[5].get_facecolor() == pytest.approx(mcolors.to_rgba("#e8e8e8"))


def test_text_and_colour_are_separate_tables():
    vals = np.array([[10.0, 10.0], [20.0, 5.0]])
    ax = nf.scorecard(
        vals,
        row_labels=["a", "b"],
        col_labels=["x", "y"],
        color_values=[[0.0, 0.0], [-30.0, 30.0]],
        text=[["one", "two"], ["red", "blue"]],
    )
    cells = _cells(ax)
    assert cells[2].get_facecolor() == pytest.approx(mcolors.to_rgba("#C4372D"))
    assert cells[3].get_facecolor() == pytest.approx(mcolors.to_rgba("#437AB5"))
    printed = {t.get_text() for t in ax.texts}
    assert {"one", "two", "red", "blue"} <= printed
    assert "20" not in printed


def test_a_cell_with_nan_colour_is_grey_even_if_it_has_a_value():
    ax = nf.scorecard(np.array([[3.0]]), row_labels=["a"], col_labels=["x"], color_values=[[np.nan]])
    assert _cells(ax)[0].get_facecolor() == pytest.approx(mcolors.to_rgba("#e8e8e8"))
    assert "3" in {t.get_text() for t in ax.texts}


# ------------------------------------------------------------------ layout
def test_row_groups_are_separated_by_a_gap_and_the_baseline_is_its_own_group():
    ax = _card([[1.0], [0.9], [0.8]], reference_row="ref")
    ys = [p.get_xy()[1] for p in _cells(ax)]
    assert ys == pytest.approx([0.0, 1.35, 2.35])
    ax2 = _card([[1.0], [0.9], [0.8], [0.7]], row_groups=["g", "g", "h", "h"], row_group_gap=0.5)
    assert [p.get_xy()[1] for p in _cells(ax2)] == pytest.approx([0.0, 1.0, 2.5, 3.5])


def test_column_groups_get_a_title_and_a_1p6pt_rule():
    ax = _card([[1.0, 2.0, 3.0]], col_groups=["G1", "G1", "G2"])
    assert sorted(t.get_text() for t in ax.texts if t.get_text() in ("G1", "G2")) == ["G1", "G2"]
    assert [ln.get_linewidth() for ln in ax.lines] == [1.6, 1.6]
    ax2 = _card([[1.0, 2.0, 3.0]], col_groups=["G1", "G1", "G2"], group_rule=False)
    assert len(ax2.lines) == 0


def test_cells_are_square_at_cell_in_and_default_to_0p42_inches():
    ax = _card([[1.0, 2.0], [1.0, 2.0]])
    fig = ax.figure
    fig.canvas.draw()
    box = ax.get_window_extent()
    assert box.width / fig.dpi == pytest.approx(2 * 0.42)
    assert box.height / fig.dpi == pytest.approx(2 * 0.42)
    ax3 = _card([[1.0, 2.0]], cell_in=0.6)
    ax3.figure.canvas.draw()
    assert ax3.get_window_extent().width / ax3.figure.dpi == pytest.approx(1.2)


def test_width_solves_the_cell_size_and_conflicts_with_cell_in():
    ax = _card([[1.0, 2.0, 3.0, 4.0]], width=4.0)
    assert ax.figure.get_figwidth() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="width or cell_in"):
        _card([[1.0]], width=4.0, cell_in=0.4)


def test_column_headings_wrap_horizontally_with_the_unit_below():
    ax = nf.scorecard(
        np.ones((1, 2)),
        row_labels=["a"],
        col_labels=["surface humidity", "ship"],
        col_units={"surface humidity": "ppm", "ship": "ppm"},
    )
    heads = {t.get_text(): t for t in ax.texts}
    assert "surface\nhumidity" in heads
    assert heads["ship"].get_rotation() == 0
    assert [t.get_text() for t in ax.texts].count("ppm") == 2


def test_text_overlaps_are_clean_for_a_grouped_card_with_colourbar():
    ax = _card(
        [[1.0, 2.0, 3.0, 4.0], [0.9, 2.2, 2.0, np.nan], [1.1, 1.9, 3.1, 4.4]],
        reference_row="ref",
        col_groups=["In situ", "In situ", "Gradient", "Column"],
        col_units={"c0": "ppm", "c1": "ppm", "c2": "ppm", "c3": "ppm"},
        colorbar=True,
        title="Skill",
    )
    assert nf.check_text_overlaps(ax.figure) == []


def test_the_baseline_example_renders_without_text_problems_and_has_the_deck_look():
    path = Path(__file__).parent.parent / "examples" / "34_scorecard_baseline.py"
    spec = importlib.util.spec_from_file_location("ex34", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig = mod.make()
    assert nf.check_text_overlaps(fig) == []
    ax = fig.axes[0]
    assert sum(1 for p in ax.patches if p.get_fill()) == 36
    assert len(_outlines(ax)) == 1
    assert "n/a" not in {t.get_text() for t in ax.texts}
    assert [a.get_xlabel() for a in fig.axes[1:]] == ["% better than the baseline"]


# ------------------------------------------------------------------ colourbar
def test_colorbar_has_band_edge_ticks_pointed_ends_and_default_label():
    ax = _card([[1.0, 2.0], [0.8, 2.5]], reference_row="ref", colorbar=True)
    cax = [a for a in ax.figure.axes if a is not ax][0]
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in cax.get_xticklabels()]
    assert labels == [f"{e:g}".replace("-", "−") for e in EDGES]
    assert cax.get_xlabel() == "% better than the baseline"


# ---------------------------------------------------------------- compatibility
def test_reference_with_uncertainty_still_scores_in_standard_errors():
    vals = np.array([[1.0], [0.0]])
    ax = nf.scorecard(
        vals,
        row_labels=["a", "b"],
        col_labels=["x"],
        direction={"x": "lower_better"},
        reference=[[1.0]],
        uncertainty=[[0.25]],  # gap of 4 sigma saturates the scale at the pointed-end colour
    )
    assert _cells(ax)[1].get_facecolor() == pytest.approx(mcolors.to_rgba("#26549A"))


def test_explicit_norm_still_colours_the_raw_values_and_rank_mode_remains_the_fallback():
    cm, norm = nf.scorecard_diverging_cmap()
    ax = nf.scorecard(np.array([[-35.0, 60.0]]), row_labels=["m"], col_labels=["a", "b"], cmap=cm, norm=norm)
    got = [p.get_facecolor() for p in _cells(ax)]
    assert got == [pytest.approx(cm(norm(-35.0))), pytest.approx(cm(norm(60.0)))]
    ranked = nf.scorecard(np.array([[1.0], [3.0]]), row_labels=["lo", "hi"], col_labels=["s"])
    lo, hi = (p.get_facecolor() for p in _cells(ranked))
    assert hi[2] > hi[0]  # best (highest) cell is blue
    assert lo[0] > lo[2]  # worst is red
    assert _outlines(ranked) == []  # highlight_best defaults to False


def test_all_figures_close():
    plt.close("all")
