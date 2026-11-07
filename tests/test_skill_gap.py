"""Contract of `skill_gap`: a signed, standardised distance from a reference.

The laws here are what let one scorecard carry metrics in different units. A raw table of
gaps cannot: an energy-score gap in PgC/yr and a correlation gap in nothing share no scale,
so any single colour ramp over them ranks the metrics by unit rather than by skill.
"""

from __future__ import annotations

import numpy as np
import pytest

from nicefigs import skill_gap


def test_a_value_equal_to_its_reference_scores_zero_whichever_way_it_is_read():
    """Zero is the fixed point of the whole idea: indistinguishable from the reference."""
    for direction in ("higher_better", "lower_better", "target"):
        gap = skill_gap([[2.0, -3.0]], [[2.0, -3.0]], direction=direction, target=0.0)
        assert np.allclose(gap, 0.0)


def test_positive_is_better_for_a_metric_read_upward():
    assert skill_gap([[0.8]], [[0.5]], direction="higher_better")[0, 0] == pytest.approx(0.3)
    assert skill_gap([[0.2]], [[0.5]], direction="higher_better")[0, 0] == pytest.approx(-0.3)


def test_positive_is_better_for_a_metric_read_downward_too():
    """The sign convention is the point: one colour scale must mean one thing everywhere."""
    assert skill_gap([[1.0]], [[3.0]], direction="lower_better")[0, 0] == pytest.approx(2.0)
    assert skill_gap([[5.0]], [[3.0]], direction="lower_better")[0, 0] == pytest.approx(-2.0)


def test_a_target_metric_is_scored_by_closeness_to_its_target_not_to_zero():
    """A spread-skill ratio is best at one; scoring it as a distance from zero calls an
    under-dispersed ensemble good, which is the failure this argument exists to prevent."""
    # value 0.8 and reference 1.15 against a target of 1: the value is *further* from target
    gap = skill_gap([[0.8]], [[1.15]], direction="target", target=1.0)[0, 0]
    assert gap == pytest.approx(abs(1.15 - 1.0) - abs(0.8 - 1.0))
    assert gap < 0
    # and scoring the same numbers against zero would have called it better
    assert skill_gap([[0.8]], [[1.15]], direction="target", target=0.0)[0, 0] > 0


def test_target_defaults_to_zero_so_a_bias_needs_no_extra_argument():
    assert skill_gap([[-0.1]], [[0.4]], direction="target")[0, 0] == pytest.approx(0.3)


def test_dividing_by_an_uncertainty_makes_metrics_in_different_units_comparable():
    """Two metrics whose gaps differ by three orders of magnitude, each one sigma from its
    own reference, must land on the same cell value."""
    gap = skill_gap(
        [[1.0e3, 1.0e-3]],
        [[0.0, 0.0]],
        uncertainty=[[1.0e3, 1.0e-3]],
        direction="higher_better",
    )
    assert np.allclose(gap, 1.0)


def test_a_zero_or_absent_uncertainty_gives_a_blank_rather_than_a_division_by_zero():
    gap = skill_gap([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]], uncertainty=[[2.0, 0.0, np.nan]])
    assert gap[0, 0] == pytest.approx(0.5)
    assert np.isnan(gap[0, 1])
    assert np.isnan(gap[0, 2])


def test_direction_may_be_given_per_column():
    gap = skill_gap(
        [[1.0, 1.0]],
        [[3.0, 3.0]],
        direction=["lower_better", "higher_better"],
    )
    assert gap[0, 0] == pytest.approx(2.0)
    assert gap[0, 1] == pytest.approx(-2.0)


def test_a_per_column_direction_must_match_the_columns_it_labels():
    with pytest.raises(ValueError, match="one direction per column"):
        skill_gap([[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0]], direction=["lower_better"])


def test_an_unknown_direction_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown direction"):
        skill_gap([[1.0]], [[0.0]], direction="bigger_is_nicer")


def test_a_missing_value_or_reference_stays_missing_rather_than_scoring_zero():
    """A metric that could not be computed is a blank cell, never an average one."""
    gap = skill_gap([[np.nan, 1.0]], [[1.0, np.nan]])
    assert np.isnan(gap).all()


def test_shapes_that_cannot_broadcast_say_so_rather_than_guessing():
    with pytest.raises(ValueError, match="same shape"):
        skill_gap([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


def test_a_single_column_reference_broadcasts_across_the_columns_it_serves():
    """One shared baseline per row is the common case and must not need tiling by the caller."""
    gap = skill_gap([[2.0, 4.0]], [[1.0]], direction="higher_better")
    assert np.allclose(gap, [[1.0, 3.0]])


def test_it_accepts_a_one_dimensional_table_as_a_single_row():
    gap = skill_gap([1.0, 2.0], [0.0, 0.0], direction="higher_better")
    assert gap.shape == (1, 2)


def test_a_scalar_reference_and_uncertainty_broadcast_over_the_table():
    gap = skill_gap([[2.0, 4.0]], 0.0, uncertainty=2.0, direction="higher_better")
    assert np.allclose(gap, [[1.0, 2.0]])


# ---------------------------------------------------------------------------
# scorecard(reference=...)
# ---------------------------------------------------------------------------


def test_a_scorecard_given_a_reference_still_prints_the_raw_value():
    """The number a reader quotes must stay the metric, not its standardised gap."""
    import matplotlib

    matplotlib.use("Agg")
    from nicefigs import scorecard

    ax = scorecard(
        [[2.5], [9.0]],
        row_labels=["a", "b"],
        col_labels=["energy score"],
        direction={"energy score": "lower_better"},
        reference=[[2.0], [2.0]],
        cell_format="{:.3g}",
    )
    printed = {t.get_text() for t in ax.texts}
    assert "2.5" in printed and "9" in printed


def test_a_scorecard_colours_by_the_gap_so_two_units_share_one_scale():
    """Two columns whose values differ by three orders of magnitude but which sit the same
    distance from their own references must be coloured alike."""
    import matplotlib

    matplotlib.use("Agg")
    from nicefigs import scorecard

    ax = scorecard(
        [[1.0e3, 1.0e-3]],
        row_labels=["a"],
        col_labels=["big", "small"],
        reference=[[0.0, 0.0]],
        uncertainty=[[1.0e3, 1.0e-3]],
    )
    faces = [p.get_facecolor() for p in ax.patches if p.get_width() == 1]
    assert faces[0] == pytest.approx(faces[1])


def test_a_scorecard_reference_respects_a_target_metric():
    """An under-dispersed ratio must not be coloured better than a calibrated one."""
    import matplotlib

    matplotlib.use("Agg")
    from nicefigs import scorecard

    ax = scorecard(
        [[0.8]],
        row_labels=["a"],
        col_labels=["spread-skill"],
        direction={"spread-skill": "target"},
        reference=[[1.15]],
        target=1.0,
    )
    assert ax is not None
