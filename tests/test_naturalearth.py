"""Maps must render from bundled Natural Earth data with the network blocked."""

from __future__ import annotations

import shutil
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nicefigs as nf
from nicefigs import maps, naturalearth

cartopy = pytest.importorskip("cartopy")
pytest.importorskip("shapely")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Empty cartopy caches and every download attempt raising, as on a compute node."""

    def _blocked(*args, **kwargs):
        raise urllib.error.URLError("network blocked")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setitem(cartopy.config, "data_dir", tmp_path / "cartopy")
    monkeypatch.setitem(cartopy.config, "pre_existing_data_dir", tmp_path / "none")
    monkeypatch.delenv(naturalearth.ENV_VAR, raising=False)
    maps._land_mask.cache_clear()
    downloader = cartopy.config["downloaders"][("shapefiles", "natural_earth")]
    monkeypatch.setattr(downloader, "_urlopen", _blocked, raising=False)
    return tmp_path


@pytest.mark.parametrize("resolution", ["110m", "50m"])
def test_map_features_render_offline(offline, resolution):
    lon, lat, v = nf.data.global_points(n=500)
    fig, ax, _ = nf.map(lon, lat, v, inset_hist=False)
    nf.draw_background(ax, maps._try_cartopy(), ocean=True, borders=True, resolution=resolution)
    fig.canvas.draw()
    assert len(ax.collections) > 0
    plt.close(fig)


def test_clip_ocean_land_mask_offline(offline):
    mask = maps._land_mask(36, 18, 10.0, "110m")
    assert mask is not None and 0.1 < mask.mean() < 0.5


def test_explicit_cartopy_dir_wins(offline):
    src = naturalearth._BUNDLED / "natural_earth" / "physical"
    dst = offline / "mine" / "shapefiles" / "natural_earth" / "physical"
    dst.mkdir(parents=True)
    for f in src.glob("ne_110m_land.*"):
        shutil.copy(f, dst)
    cartopy.config["pre_existing_data_dir"] = offline / "mine"
    maps._try_cartopy()
    from cartopy.io.shapereader import natural_earth

    path = natural_earth(resolution="110m", category="physical", name="land")
    assert offline / "mine" in path.parents


def test_env_dir_beats_bundled_for_missing_resolution(offline, monkeypatch):
    dst = offline / "ne" / "shapefiles" / "natural_earth" / "physical"
    dst.mkdir(parents=True)
    src = naturalearth._BUNDLED / "natural_earth" / "physical"
    for f in src.glob("ne_110m_land.*"):
        shutil.copy(f, dst / f.name.replace("110m", "10m"))
    monkeypatch.setenv(naturalearth.ENV_VAR, str(offline / "ne"))
    maps._try_cartopy()
    from cartopy.io.shapereader import natural_earth

    assert offline / "ne" in natural_earth(resolution="10m", category="physical", name="land").parents


def test_cache_natural_earth_requests_every_layer(offline, monkeypatch):
    import cartopy.io.shapereader as sr

    seen = []
    monkeypatch.setattr(sr, "natural_earth", lambda **kw: seen.append(kw))
    out = nf.cache_natural_earth(offline / "cache", resolutions=["10m"])
    assert out == offline / "cache"
    assert {(k["category"], k["name"]) for k in seen} == set(naturalearth.LAYERS)
    assert {k["resolution"] for k in seen} == {"10m"}
    assert cartopy.config["data_dir"] == offline / "cartopy"  # restored
    with pytest.raises(ValueError):
        nf.cache_natural_earth(offline, resolutions=["5m"])


def _stipple(style, **kw):
    lon, lat, v = nf.data.global_points(n=1500)
    fig, ax, _ = nf.map(lon, lat, v, inset_hist=False)
    cs = nf.hatch_region(
        ax, lon, lat, (np.abs(lat) < 30).astype(float), ccrs=maps._try_cartopy(), bin_deg=6, style=style, **kw
    )
    return fig, cs


def test_hatch_region_dots_have_no_outline():
    fig, cs = _stipple("dots", edgecolor="#cc0000")
    assert list(cs.hatches) == ["..."]
    assert cs.get_linewidth()[0] == 0.0
    assert len(cs.get_edgecolor()) == 0 or cs.get_edgecolor()[0][3] == 0.0
    fig.canvas.draw()
    plt.close(fig)


def test_hatch_region_default_style_unchanged():
    fig, cs = _stipple("hatch")
    assert list(cs.hatches) == ["////"]
    assert cs.get_linewidth()[0] == 0.5
    plt.close(fig)


def test_hatch_region_rejects_unknown_style():
    with pytest.raises(ValueError, match="style"):
        _stipple("stripes")
