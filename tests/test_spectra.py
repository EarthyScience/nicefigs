"""Contract tests for eigen-spectrum plots."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import nicefigs as nf


def _curves(ax):
    """Data curves: solid/dashed lines that are not threshold-marker sets."""
    return [ln for ln in ax.lines if ln.get_linestyle() != "None" and not ln.get_label().startswith("_")]


def _markers(ax):
    return [ln for ln in ax.lines if ln.get_linestyle() == "None"]


def test_one_curve_per_series_on_log_log_axes():
    rng = np.random.default_rng(0)
    spectra = {f"lead {h} h": np.sort(rng.gamma(1.0, size=64))[::-1] for h in (6, 24, 72)}
    ax = nf.spectrum(spectra)
    assert len(_curves(ax)) == 3
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert [ln.get_label() for ln in _curves(ax)] == list(spectra)


def test_single_array_and_sequence_inputs():
    lam = 1.0 / np.arange(1, 33)
    assert len(_curves(nf.spectrum(lam))) == 1
    ax = nf.spectrum([lam, lam[:16] ** 2], labels=["raw", "re-normed"])
    assert [ln.get_label() for ln in _curves(ax)] == ["raw", "re-normed"]


def test_normalized_curve_sums_to_one_with_ranks_from_one():
    lam = np.array([5.0, 3.0, 1.5, 0.5])
    (ln,) = _curves(nf.spectrum(lam))
    np.testing.assert_allclose(ln.get_xdata(), [1, 2, 3, 4])
    np.testing.assert_allclose(np.sum(ln.get_ydata()), 1.0)
    (raw,) = _curves(nf.spectrum(lam, normalize=False))
    np.testing.assert_allclose(raw.get_ydata(), lam)


def test_non_positive_tail_dropped_without_warnings():
    lam = np.array([4.0, 2.0, 1.0, 0.0, -1e-12, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = nf.spectrum(lam)
        ax.figure.canvas.draw()
    (ln,) = _curves(ax)
    np.testing.assert_allclose(ln.get_xdata(), [1, 2, 3])
    assert np.all(ln.get_ydata() > 0)


def test_unsorted_input_is_plotted_descending():
    (ln,) = _curves(nf.spectrum(np.array([1.0, 4.0, 2.0])))
    assert np.all(np.diff(ln.get_ydata()) <= 0)


def test_cumulative_ends_at_one_on_log_x_linear_y():
    lam = 1.0 / np.arange(1, 101) ** 1.5
    ax = nf.spectrum({"a": lam, "b": lam[:50]}, cumulative=True)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "linear"
    for ln in _curves(ax):
        y = np.asarray(ln.get_ydata())
        assert np.all(np.diff(y) >= 0)
        assert y[-1] == pytest.approx(1.0)


def test_threshold_crossings_at_first_rank_reaching_each_threshold():
    # cumulative fractions 0.4, 0.7, 0.9, 1.0
    lam = np.array([4.0, 3.0, 2.0, 1.0])
    ax = nf.spectrum(lam, cumulative=True, thresholds=(0.5, 0.9, 0.95))
    (mk,) = _markers(ax)
    np.testing.assert_allclose(mk.get_xdata(), [2, 3, 4])
    np.testing.assert_allclose(mk.get_ydata(), [0.7, 0.9, 1.0])
    guides = [ln for ln in ax.lines if ln not in _curves(ax) and ln not in _markers(ax)]
    assert sorted(g.get_ydata()[0] for g in guides) == pytest.approx([0.5, 0.9, 0.95])


def test_reference_power_law_slope():
    lam = 1.0 / np.arange(1, 65)
    ax = nf.spectrum(lam, reference_alpha=2.0)
    ref = [ln for ln in ax.lines if ln.get_label().startswith("$k^")]
    assert len(ref) == 1
    x, y = np.asarray(ref[0].get_xdata()), np.asarray(ref[0].get_ydata())
    slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    assert slope == pytest.approx(-2.0)


def test_draws_into_given_axes():
    _, ax = nf.subplots()
    assert nf.spectrum(np.ones(4), ax=ax) is ax


def test_reference_with_cumulative_raises():
    with pytest.raises(ValueError):
        nf.spectrum(np.ones(4), cumulative=True, reference_alpha=1.0)


def test_round_off_tail_below_rtol_dropped():
    lam = np.array([1.0, 0.1, 1e-3, 1e-17, 1e-19])
    (ln,) = _curves(nf.spectrum(lam, normalize=False))
    np.testing.assert_allclose(ln.get_ydata(), lam[:3])
    (ln,) = _curves(nf.spectrum(lam, normalize=False, rtol=0.0))
    assert len(ln.get_xdata()) == 5


@pytest.mark.parametrize("normalize", [True, False])
def test_reference_passes_through_the_plotted_leading_value(normalize):
    lam = np.array([8.0, 4.0, 2.0, 1.0])
    ax = nf.spectrum({"a": lam, "b": lam / 2}, reference_alpha=1.0, normalize=normalize)
    (ref,) = [ln for ln in ax.lines if ln.get_label().startswith("$k^")]
    lead = _curves(ax)[0].get_ydata()[0]
    assert ref.get_xdata()[0] == pytest.approx(1.0)
    assert ref.get_ydata()[0] == pytest.approx(lead)


def test_threshold_guides_labelled_once_whatever_the_number_of_series():
    lam = 1.0 / np.arange(1, 41)
    ax = nf.spectrum({"a": lam, "b": lam**2, "c": lam**0.5}, cumulative=True, thresholds=(0.5, 0.9))
    assert sorted(t.get_text().strip() for t in ax.texts) == ["50%", "90%"]
    assert len(_markers(ax)) == 3


def test_eigen_spectra_is_deterministic_and_descending():
    a, b = nf.data.eigen_spectra(), nf.data.eigen_spectra()
    assert list(a) == list(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert np.all(np.diff(a[name]) <= 1e-12)
