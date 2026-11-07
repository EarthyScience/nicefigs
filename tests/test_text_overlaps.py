"""figure.check_text_overlaps: overlapping and clipped text, as the renderer draws it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import nicefigs as nf


@pytest.fixture
def fig():
    f, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    yield f
    plt.close(f)


def _texts(issues, kind):
    return [i.texts for i in issues if i.kind == kind]


def test_a_tidy_figure_has_no_issues(fig):
    ax = fig.axes[0]
    ax.text(1, 8, "left")
    ax.text(7, 2, "right")
    ax.set_xlabel("x")
    fig.tight_layout()
    assert nf.check_text_overlaps(fig) == []


def test_two_overlapping_texts_are_reported_once(fig):
    ax = fig.axes[0]
    ax.text(5, 5, "first label")
    ax.text(5.3, 5.05, "second label")
    assert _texts(nf.check_text_overlaps(fig), "overlap") == [("first label", "second label")]


def test_text_outside_the_figure_is_clipped(fig):
    fig.axes[0].text(1.3, 0.5, "far right", transform=fig.axes[0].transAxes)
    issues = nf.check_text_overlaps(fig)
    assert _texts(issues, "clipped") == [("far right",)]
    assert nf.check_text_overlaps(fig, clipped=False) == []


def test_tick_labels_outside_the_view_are_not_counted(fig):
    ax = fig.axes[0]
    ax.set_xticks([-50, 5, 50])  # two ticks fall outside the limits and are never drawn
    ax.set_xticklabels(["hidden a", "shown", "hidden b"])
    ax.text(5, -0.2, "x", transform=ax.get_xaxis_transform())
    texts = [t for i in nf.check_text_overlaps(fig) for t in i.texts]
    assert "hidden a" not in texts and "hidden b" not in texts


def test_crowded_tick_labels_count_unless_ignored(fig):
    ax = fig.axes[0]
    ax.set_xticks(range(11))
    ax.set_xticklabels([f"long tick label {i}" for i in range(11)])
    assert _texts(nf.check_text_overlaps(fig), "overlap")
    assert nf.check_text_overlaps(fig, ignore_ticks=True, clipped=False) == []
