"""Role palette: colours by role, plot keywords, colour-vision-deficiency separation."""

from __future__ import annotations

import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf

ROLE_NAMES = ("primary", "contrast", "baseline", "null", "reference")

# Machado, Oliveira & Fernandes (2009) CVD simulation at severity 1.0, applied in linear sRGB.
_MACHADO = {
    "protan": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "deutan": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    "tritan": [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}


def _linear_rgb(color: str) -> np.ndarray:
    c = np.asarray(mcolors.to_rgb(color))
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(rgb: np.ndarray) -> np.ndarray:
    lms = np.cbrt(
        np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )
        @ rgb
    )
    return (
        np.array(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        )
        @ lms
    )


def _delta_e(a: str, b: str, vision: str | None) -> float:
    """OKLab distance x100 between two colours as seen under ``vision`` (None: normal)."""
    ra, rb = _linear_rgb(a), _linear_rgb(b)
    if vision is not None:
        m = np.asarray(_MACHADO[vision])
        ra, rb = np.clip(m @ ra, 0, 1), np.clip(m @ rb, 0, 1)
    return 100 * float(np.linalg.norm(_oklab(ra) - _oklab(rb)))


def test_roles_cover_the_five_named_roles():
    assert set(ROLE_NAMES) <= set(nf.ROLES)


def test_role_colors_are_the_documented_values():
    expected = {
        "primary": "#4477AA",
        "contrast": "#D95F02",
        "baseline": "#3A3A3A",
        "null": "#AAAAAA",
        "reference": "#000000",
    }
    for role, hex_ in expected.items():
        assert mcolors.same_color(nf.role_color(role), hex_)
    assert mcolors.same_color(nf.role_color("null", band=True), "#E6E6E6")


@pytest.mark.parametrize("vision", [None, "protan", "deutan", "tritan"])
def test_every_pair_of_roles_stays_apart_under_colour_vision_deficiency(vision):
    colors = [nf.role_color(r) for r in ROLE_NAMES]
    worst = min(_delta_e(a, b, vision) for a, b in itertools.combinations(colors, 2))
    assert worst >= 16.0


def test_null_band_sits_between_the_null_colour_and_white():
    null, band = nf.role_color("null"), nf.role_color("null", band=True)
    assert _delta_e(band, "#FFFFFF", None) > 5.0
    assert _delta_e(band, null, None) > 15.0


def test_unknown_role_names_the_available_roles():
    with pytest.raises(KeyError, match="primary"):
        nf.role_color("sae")
    with pytest.raises(KeyError, match="primary"):
        nf.role_style("sae")


def test_band_requested_for_a_role_without_one_raises():
    with pytest.raises(ValueError, match="band"):
        nf.role_color("primary", band=True)
    with pytest.raises(ValueError, match="band"):
        nf.role_style("primary", "band")


@pytest.mark.parametrize("role", ROLE_NAMES)
@pytest.mark.parametrize("kind", ["marker", "line", "both"])
def test_role_style_draws_with_plot_and_errorbar(role, kind):
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1, 2], [1, 2, 3], **nf.role_style(role, kind))
    ax.errorbar([0, 1], [1, 2], yerr=[0.1, 0.2], **nf.role_style(role, kind))
    assert mcolors.same_color(line.get_color(), nf.role_color(role))


def test_marker_style_draws_markers_only_and_line_style_lines_only():
    fig, ax = plt.subplots()
    (m,) = ax.plot([0, 1], [0, 1], **nf.role_style("primary", "marker"))
    (ln,) = ax.plot([0, 1], [0, 1], **nf.role_style("primary", "line"))
    no_marker = ("", "none", "None", None)
    assert m.get_linestyle() == "None" and m.get_marker() not in no_marker
    assert ln.get_linestyle() != "None" and ln.get_marker() in no_marker


def test_null_markers_are_open_and_data_markers_filled():
    fig, ax = plt.subplots()
    (null,) = ax.plot([0], [0], **nf.role_style("null"))
    (sae,) = ax.plot([0], [0], **nf.role_style("primary"))
    assert null.get_markerfacecolor() == "none"
    assert mcolors.same_color(sae.get_markerfacecolor(), nf.role_color("primary"))


def test_open_variant_keeps_the_role_colour():
    fig, ax = plt.subplots()
    (ln,) = ax.plot([0], [0], **nf.role_style("primary", filled=False))
    assert ln.get_markerfacecolor() == "none"
    assert mcolors.same_color(ln.get_markeredgecolor(), nf.role_color("primary"))


def test_roles_share_one_marker_shape_only_within_a_role():
    fig, ax = plt.subplots()
    shapes = {
        r: ax.plot([0], [0], **nf.role_style(r))[0].get_marker() for r in ("primary", "contrast", "baseline")
    }
    assert len(set(shapes.values())) == 3


def test_null_and_reference_draw_behind_the_data():
    data = nf.role_style("primary")["zorder"]
    assert nf.role_style("null")["zorder"] < data
    assert nf.role_style("reference", "line")["zorder"] < data
    assert nf.role_style("null", "band")["zorder"] < nf.role_style("null")["zorder"]


def test_reference_line_is_thinner_and_dashed():
    fig, ax = plt.subplots()
    (ref,) = ax.plot([0, 1], [0, 1], **nf.role_style("reference", "line"))
    (sae,) = ax.plot([0, 1], [0, 1], **nf.role_style("primary", "line"))
    assert ref.get_linestyle() == "--"
    assert ref.get_linewidth() < sae.get_linewidth()


def test_line_width_follows_the_active_theme():
    with nf.style("paper") as th:
        paper = nf.role_style("primary", "line")["linewidth"]
        assert paper == pytest.approx(th.lw_line)
    with nf.style("talk"):
        talk = nf.role_style("primary", "line")["linewidth"]
    assert talk > paper


def test_band_style_fills_behind_a_curve():
    fig, ax = plt.subplots()
    poly = ax.fill_between([0, 1, 2], [0, 0, 0], [1, 1, 1], **nf.role_style("null", "band"))
    assert mcolors.same_color(poly.get_facecolor()[0], nf.role_color("null", band=True))


def test_overrides_win_over_role_defaults():
    kw = nf.role_style("primary", "marker", markersize=4.5, zorder=5)
    assert kw["markersize"] == 4.5 and kw["zorder"] == 5
