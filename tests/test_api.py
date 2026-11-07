"""Import surface: everything in ``__all__`` exists and is usable."""

from __future__ import annotations

import inspect
from importlib.metadata import version

import nicefigs as nf


def test_version_matches_installed_metadata():
    assert nf.__version__ == version("nicefigs")


def test_all_names_present():
    missing = [name for name in nf.__all__ if not hasattr(nf, name)]
    assert missing == []


def test_all_names_are_public_symbols():
    for name in nf.__all__:
        obj = getattr(nf, name)
        assert obj is not None or name == "DEFAULTS"


def test_callables_are_callable():
    non_callable_allowed = {
        "THEMES",
        "COLORS",
        "PALETTE_BRIGHT",
        "PALETTE_MUTED",
        "PALETTE_HIGH_CONTRAST",
        "CATEGORICAL",
        "DEFAULTS",
        "TERNARY_PALETTES",
        "METRIC_NAMES",
        "METRIC_DIRECTION",
        "CUSTOM_CMAPS",
        "ARTIST_PALETTES",
        "HOCKNEY_CANYON",
        "NICE",
        "__version__",
        "data",
        "SCORECARD_DIVERGING_BOUNDARIES",
        "DIRECTIONS",
        "ROLES",
    }
    for name in nf.__all__:
        if name in non_callable_allowed:
            continue
        obj = getattr(nf, name)
        assert callable(obj) or inspect.isclass(obj), f"{name} should be callable"


def test_data_submodule_has_generators():
    for fn in (
        "global_points",
        "regional_grid",
        "elevation_field",
        "landcover_patch",
        "rgb_patch",
        "heatmap_matrix",
        "model_scores",
        "timeseries",
        "xy_agreement",
        "ternary_points",
        "scatter_bubbles",
        "map_timeseries_cube",
    ):
        assert hasattr(nf.data, fn)
        assert callable(getattr(nf.data, fn))
