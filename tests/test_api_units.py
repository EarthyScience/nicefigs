"""Contracts of metric labels and direction arrows."""

from __future__ import annotations

import pytest

import nicefigs as nf


@pytest.mark.parametrize(
    ("direction", "arrow"),
    [("lower_better", "↓"), ("higher_better", "↑"), ("target", ""), (None, "")],
)
def test_axis_label_appends_direction_arrow(direction, arrow):
    label = nf.axis_label("RMSE", "K", direction=direction)
    assert label == f"RMSE [K] {arrow}".rstrip()


@pytest.mark.parametrize("direction", ["down", "↓", "zero_better"])
def test_axis_label_rejects_unknown_direction(direction):
    with pytest.raises(ValueError, match="unknown direction"):
        nf.axis_label("RMSE", "K", direction=direction)


def test_metric_label_is_case_insensitive_and_direction_false_omits_arrow():
    assert nf.metric_label("R2") == "R²"
    assert nf.metric_label("rmse", direction=False) == "RMSE"


def test_metric_label_custom_registry():
    assert nf.metric_label("acc", registry={"acc": "Top-1"}) == "Top-1"
