"""units.py: unit prettification, axis labels, numeric formatting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from nicefigs.units import axis_label, fmt_pval, fmt_val, prettify_unit, set_labels


def test_prettify_unit_simple_exponent():
    assert prettify_unit("m2") == "m²"


def test_prettify_unit_compound_exponents():
    assert prettify_unit("kg m-2 s-1") == "kg m⁻² s⁻¹"


def test_prettify_unit_degc():
    assert prettify_unit("degC") == "°C"


def test_prettify_unit_deg_word_replacement():
    assert prettify_unit("deg") == "°"


def test_prettify_unit_percent():
    assert prettify_unit("percent") == "%"


def test_prettify_unit_micro_prefixes():
    assert "µm" in prettify_unit("um") or prettify_unit("um") == "µm"
    assert prettify_unit("ug") == "µg"


def test_prettify_unit_none_returns_empty():
    assert prettify_unit(None) == ""


def test_prettify_unit_empty_string():
    assert prettify_unit("") == ""


def test_prettify_unit_superscript_false_uses_caret():
    out = prettify_unit("m2", superscript=False)
    assert out == "m^2"


def test_prettify_unit_caret_notation_input():
    assert prettify_unit("s^-1") == "s⁻¹"


def test_axis_label_composes_quantity_and_unit():
    assert axis_label("Temperature", "degC") == "Temperature [°C]"


def test_axis_label_no_unit_returns_quantity_only():
    assert axis_label("Temperature", None) == "Temperature"
    assert axis_label("Temperature", "") == "Temperature"


def test_axis_label_custom_brackets():
    assert axis_label("Temperature", "degC", brackets="()") == "Temperature (°C)"


def test_set_labels_sets_xlabel_ylabel_title():
    fig, ax = plt.subplots()
    try:
        set_labels(ax, "Time", "Temperature", xunits=None, yunits="degC", title="My title")
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Temperature [°C]"
        assert ax.get_title() == "My title"
    finally:
        plt.close(fig)


def test_set_labels_partial_leaves_others_untouched():
    fig, ax = plt.subplots()
    try:
        ax.set_xlabel("preexisting")
        set_labels(ax, y="Value")
        assert ax.get_xlabel() == "preexisting"
        assert ax.get_ylabel() == "Value"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "value,expected_contains",
    [
        (0.0, "0"),
        (123.456, "123"),
        (-5.5, "-5.5"),
    ],
)
def test_fmt_val_common_cases(value, expected_contains):
    assert expected_contains in fmt_val(value)


def test_fmt_val_zero_is_exactly_zero_string():
    assert fmt_val(0.0) == "0"


def test_fmt_val_nan_returns_dash():

    assert fmt_val(float("nan")) == "–"


def test_fmt_val_small_number_uses_scientific():
    s = fmt_val(0.0001234)
    assert "×10" in s
    assert s.startswith("1.2")


def test_fmt_val_large_number_uses_scientific():
    s = fmt_val(123456789.0)
    assert "×10" in s


def test_fmt_val_negative_small_scientific():
    s = fmt_val(-0.0005)
    assert s.startswith("-5")
    assert "×10" in s


def test_fmt_val_mid_range_not_scientific():
    s = fmt_val(42.123)
    assert "×10" not in s


def test_fmt_pval_very_small():
    assert fmt_pval(0.0001) == "p < 0.001"


def test_fmt_pval_small():
    assert fmt_pval(0.005) == "p = 0.005"


def test_fmt_pval_moderate():
    assert fmt_pval(0.5) == "p = 0.50"
