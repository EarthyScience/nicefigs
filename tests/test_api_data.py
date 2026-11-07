"""Contract of the synthetic example data: deterministic per seed, documented shapes."""

from __future__ import annotations

import numpy as np
import pytest

from nicefigs import data as npd

GENERATORS = [
    npd.global_points,
    npd.regional_grid,
    npd.elevation_field,
    npd.landcover_patch,
    npd.rgb_patch,
    npd.heatmap_matrix,
    npd.model_scores,
    npd.timeseries,
    npd.xy_agreement,
    npd.ternary_points,
    npd.scatter_bubbles,
    npd.pareto_tradeoff,
    npd.site_metrics,
    npd.cross_section_field,
    npd.global_grid,
    npd.icosahedral_field,
    npd.map_timeseries_cube,
    npd.taylor_stats,
    npd.hovmoller_field,
    npd.vertical_profiles,
]


def _equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and not np.isscalar(a):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.__name__)
def test_generator_is_deterministic_per_seed(gen):
    assert _equal(gen(seed=1), gen(seed=1))
    assert not _equal(gen(seed=1), gen(seed=2))


def test_global_points_returns_aligned_degrees_and_rejects_unknown_field():
    lon, lat, vals = npd.global_points(500, land_bias=False)
    assert lon.shape == lat.shape == vals.shape == (500,)
    assert np.all((-180 <= lon) & (lon <= 180)) and np.all((-90 <= lat) & (lat <= 90))
    assert npd.global_points(500, field="sparse")[2].min() >= 0
    with pytest.raises(ValueError, match="field"):
        npd.global_points(field="pressure")


def test_ternary_compositions_sum_to_one():
    comp, values = npd.ternary_points(n=50)
    assert comp.shape == (50, 3) and values.shape == (50,)
    np.testing.assert_allclose(comp.sum(1), 1.0)


def test_taylor_stats_crmsd_is_consistent_with_law_of_cosines():
    for s in npd.taylor_stats(models=["a", "b"]):
        expected = np.sqrt(1 + s["std_ratio"] ** 2 - 2 * s["std_ratio"] * s["correlation"])
        assert s["crmsd"] == pytest.approx(expected)


def test_global_grid_shape_matches_coordinates():
    lon, lat, z = npd.global_grid(res=10.0)
    assert z.shape == (lat.size, lon.size)
    assert lat[0] > lat[-1]
