"""Deterministic synthetic example data for the gallery, docs and tests.

Every generator is numpy-only and reproducible from ``seed``. Fields look
geophysical (smooth spatial structure, a land bias, seasonal cycles) but are not
real data.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _land_weight(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A smooth [0,1] field peaking over the major continents (very rough)."""
    centers = [
        (10, 50, 40),
        (100, 40, 45),
        (-100, 40, 35),
        (-60, -10, 30),
        (20, 0, 30),
        (135, -25, 25),
    ]
    w = np.zeros_like(lon, dtype=float)
    for clon, clat, r in centers:
        d2 = ((lon - clon) / r) ** 2 + ((lat - clat) / (r * 0.7)) ** 2
        w += np.exp(-d2)
    return np.clip(w, 0, 1)


def global_points(
    n: int = 4000, *, seed: int = 0, land_bias: bool = True, field: str = "temperature"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scattered global stations with a synthetic signal.

    Args:
        n: Number of candidate stations; ``land_bias`` keeps a subset of them.
        seed: Random seed.
        land_bias: Thin out stations away from the continents.
        field: ``"temperature"`` (latitude-driven, °C), ``"anomaly"``
            (zero-centred, for diverging maps) or ``"sparse"`` (non-negative,
            zero at about half the stations).

    Returns:
        ``(lon, lat, values)``, 1-D arrays of equal length; degrees.

    Raises:
        ValueError: Unknown ``field``.
    """
    rng = _rng(seed)
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-90, 90, n)
    if land_bias:
        keep = rng.uniform(0, 1, n) < (0.25 + 0.75 * _land_weight(lon, lat))
        lon, lat = lon[keep], lat[keep]
    smooth = (
        np.sin(np.deg2rad(lat) * 2)
        + 0.6 * np.cos(np.deg2rad(lon) * 1.5)
        + 0.4 * np.sin(np.deg2rad(lon + lat))
    )
    if field == "temperature":
        vals = 30 - 0.55 * np.abs(lat) + 4 * smooth + _rng(seed + 1).normal(0, 2, lat.shape)
    elif field == "anomaly":
        vals = 2.2 * smooth + _rng(seed + 1).normal(0, 0.8, lat.shape)
    elif field == "sparse":
        base = np.clip(smooth + 1.2, 0, None)
        vals = base * (_rng(seed + 2).uniform(0, 1, lat.shape) < 0.5)
    else:
        raise ValueError(f"Unknown field {field!r}; use 'temperature', 'anomaly' or 'sparse'.")
    return lon, lat, vals


def regional_grid(
    *, seed: int = 0, extent: tuple[float, float, float, float] = (5, 20, 45, 55), res: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth regional field on a lon/lat grid.

    Args:
        seed: Random seed.
        extent: ``(west, east, south, north)`` in degrees.
        res: Grid spacing in degrees.

    Returns:
        ``(lon2d, lat2d, z)``, three 2-D arrays of shape ``(nlat, nlon)``.
    """
    w, e, s, n = extent
    lon = np.arange(w, e, res)
    lat = np.arange(s, n, res)
    lon2d, lat2d = np.meshgrid(lon, lat)
    rng = _rng(seed)
    z = np.zeros_like(lon2d)
    for _ in range(6):
        cx, cy = rng.uniform(w, e), rng.uniform(s, n)
        z += rng.normal() * np.exp(-(((lon2d - cx) / 2) ** 2 + ((lat2d - cy) / 1.5) ** 2))
    return lon2d, lat2d, z


def elevation_field(*, seed: int = 3, shape: tuple[int, int] = (180, 360)) -> np.ndarray:
    """A global elevation raster in metres, negative over ocean.

    Returns:
        Array of ``shape`` ``(nlat, nlon)``, north-up, spanning -180..180°E.
    """
    rng = _rng(seed)
    ny, nx = shape
    lat = np.linspace(90, -90, ny)[:, None]
    lon = np.linspace(-180, 180, nx)[None, :]
    z = 1200 * _land_weight(lon * np.ones_like(lat), lat * np.ones_like(lon)) - 400
    for _ in range(40):
        cx, cy, a, r = rng.uniform(-180, 180), rng.uniform(-60, 70), rng.normal(0, 1500), rng.uniform(3, 20)
        z += a * np.exp(-(((lon - cx) / r) ** 2 + ((lat - cy) / r) ** 2))
    return z


def landcover_patch(*, seed: int = 5, size: int = 96, product: str = "esa_worldcover") -> np.ndarray:
    """A blobby ``(size, size)`` raster of land-cover class codes.

    Args:
        seed: Random seed.
        size: Edge length in pixels.
        product: Name of a :func:`nicefigs.categorical` scheme whose codes are used.

    Returns:
        Integer array of class codes of ``product``.
    """
    from .palettes import categorical

    codes = categorical(product).codes
    rng = _rng(seed)
    lab = np.zeros((size, size), dtype=int) + codes[0]
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(28):
        cy, cx, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(6, 22)
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
        lab[m] = rng.choice(codes)
    return lab


def rgb_patch(*, seed: int = 7, size: int = 96) -> np.ndarray:
    """A natural-colour ``(size, size, 3)`` ``uint8`` image."""
    rng = _rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    r = 0.35 + 0.3 * np.sin(6 * xx) + 0.1 * rng.normal(0, 1, (size, size))
    g = 0.5 + 0.3 * np.cos(5 * yy) + 0.1 * rng.normal(0, 1, (size, size))
    b = 0.3 + 0.2 * (xx + yy) + 0.1 * rng.normal(0, 1, (size, size))
    img = np.clip(np.stack([r, g, b], -1), 0, 1)
    return (img * 255).astype(np.uint8)


def heatmap_matrix(*, seed: int = 9, shape: tuple[int, int] = (8, 12)) -> np.ndarray:
    """A smoothly structured matrix of ``shape`` with values around ``[0, 1]``."""
    rng = _rng(seed)
    base = np.outer(np.linspace(0.2, 1, shape[0]), np.linspace(1, 0.3, shape[1]))
    return base + rng.normal(0, 0.08, shape)


def model_scores(
    *,
    seed: int = 11,
    models: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    timescales: Sequence[str] | None = None,
    n: int = 60,
) -> dict[str, Any]:
    """Per-fold scores of several models, metrics and timescales.

    Models differ in skill, and skill decays with the timescale index. Metrics
    named ``"RMSE"`` or starting with ``"R"`` get error-like and correlation-like
    values; any other metric gets bias-like values.

    Args:
        seed: Random seed.
        models: Model names; default four forecast-model names.
        metrics: Metric names; default ``RMSE``, ``R²`` and ``Bias``.
        timescales: Timescale names; default daily, weekly, monthly.
        n: Folds per cell.

    Returns:
        Dict with ``scores`` (``{metric: {model: {timescale: (n,) array}}}``)
        and the ``models``, ``metrics`` and ``timescales`` lists.
    """
    rng = _rng(seed)
    models = models or ["Persistence", "Climatology", "U-Net", "Transformer"]
    metrics = metrics or ["RMSE", "R$^2$", "Bias"]
    timescales = timescales or ["daily", "weekly", "monthly"]
    known_skill = {"Persistence": 0.35, "Climatology": 0.5, "U-Net": 0.72, "Transformer": 0.8}
    # Custom model names get a skill spread evenly over [0.4, 0.85] by order.
    skill = {
        mdl: known_skill.get(mdl, 0.4 + 0.45 * (i / max(len(models) - 1, 1))) for i, mdl in enumerate(models)
    }
    out = {}
    for m in metrics:
        out[m] = {}
        for mdl in models:
            out[m][mdl] = {}
            for ti, ts in enumerate(timescales):
                s = skill[mdl] * (1 - 0.12 * ti)
                if m == "RMSE":
                    v = (1 - s) * 2 + rng.normal(0, 0.12, n)
                elif m.startswith("R"):
                    v = s + rng.normal(0, 0.06, n)
                else:
                    v = (0.5 - s) * 0.5 + rng.normal(0, 0.08, n)
                out[m][mdl][ts] = v
    return {"scores": out, "models": models, "metrics": metrics, "timescales": timescales}


def timeseries(
    *, seed: int = 13, n_days: int = 730, series: Sequence[str] | None = None
) -> tuple[list[_dt.date], dict[str, np.ndarray]]:
    """Daily series with a seasonal cycle, a trend and noise, starting 2018-01-01.

    The first series is unbiased; later ones carry a random bias and more noise.

    Args:
        seed: Random seed.
        n_days: Length in days.
        series: Series names; default ``observed``, ``model A``, ``model B``.

    Returns:
        ``(dates, {name: (n_days,) array})``.
    """
    rng = _rng(seed)
    series = series or ["observed", "model A", "model B"]
    t = np.arange(n_days)
    seasonal = 12 * np.sin(2 * np.pi * t / 365.25 - 1.3)
    trend = 0.004 * t
    dates = [_dt.date(2018, 1, 1) + _dt.timedelta(days=int(i)) for i in t]
    data = {}
    for i, s in enumerate(series):
        bias = 0 if i == 0 else rng.normal(0, 1.5)
        amp = 1 + 0.1 * i
        data[s] = 15 + amp * seasonal + trend + bias + rng.normal(0, 1.2 + 0.5 * i, n_days)
    return dates, data


def xy_agreement(
    *, seed: int = 15, n: int = 1500, r: float = 0.9, outlier_frac: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """Paired ``(observed, predicted)`` samples with a small bias and some outliers.

    Args:
        seed: Random seed.
        n: Number of pairs.
        r: Target correlation before outliers are added.
        outlier_frac: Fraction of pairs perturbed into outliers.

    Returns:
        ``(x, y)``, two ``(n,)`` arrays.
    """
    rng = _rng(seed)
    x = rng.gamma(2.5, 2.0, n)
    noise = rng.normal(0, x.std() * np.sqrt(1 - r**2), n)
    y = r * (x - x.mean()) + x.mean() + noise + 0.6
    n_out = int(outlier_frac * n)
    idx = rng.choice(n, n_out, replace=False)
    y[idx] += rng.normal(0, x.std() * 2, n_out)
    return x, y


def ternary_points(*, seed: int = 17, n: int = 800, clusters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Clustered three-part compositions and a value per point.

    Returns:
        ``(comp, values)``: ``comp`` of shape ``(n, 3)`` with rows summing to 1,
        ``values`` of shape ``(n,)``.
    """
    rng = _rng(seed)
    centers = rng.dirichlet(np.ones(3), clusters)
    which = rng.integers(0, clusters, n)
    comp = np.array([rng.dirichlet(centers[w] * 12 + 0.5) for w in which])
    values = comp[:, 0] * 2 - comp[:, 2] + rng.normal(0, 0.1, n)
    return comp, values


def scatter_bubbles(*, seed: int = 19, n: int = 120) -> dict[str, np.ndarray]:
    """Points for a multi-encoding scatter.

    Returns:
        Dict of ``(n,)`` arrays: ``x``, ``y``, ``size`` (marker area, pt²),
        ``color`` (a continuous value) and ``group`` (``"A"``, ``"B"`` or ``"C"``).
    """
    rng = _rng(seed)
    groups = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    x = rng.uniform(0, 10, n)
    y = 0.5 * x + rng.normal(0, 1.5, n) + (groups == "B") * 2 - (groups == "C") * 1.5
    size = rng.uniform(10, 200, n)
    color = x + rng.normal(0, 1, n)
    return {"x": x, "y": y, "size": size, "color": color, "group": groups}


def pareto_tradeoff(
    *, seed: int = 23, groups: Sequence[str] | None = None, n_per: int = 9
) -> dict[str, np.ndarray]:
    """A speed-versus-skill cloud in which each group traces its own trade-off frontier.

    Args:
        seed: Random seed.
        groups: Group names; default four model-family names.
        n_per: Points per group.

    Returns:
        Dict of arrays: ``x`` (a speed, higher is better), ``y`` (a skill,
        higher is better; it falls as ``x`` rises within a group), ``group``
        (group name) and ``size`` (a model size in ``{256, 512, 1024, 2048}``).
    """
    rng = _rng(seed)
    groups = list(groups or ["Linear", "Tree", "CNN", "Transformer"])
    offset = {a: 0.12 * i for i, a in enumerate(groups)}
    x, y, group, size = [], [], [], []
    for a in groups:
        widths = rng.uniform(0.2, 1.0, n_per)
        xa = 0.15 + 0.5 * widths + offset[a] + rng.normal(0, 0.02, n_per)
        ya = 0.95 - 0.6 * widths - 0.5 * offset[a] + rng.normal(0, 0.02, n_per)
        x.extend(xa)
        y.extend(ya)
        group.extend([a] * n_per)
        size.extend(rng.choice([256, 512, 1024, 2048], n_per))
    return {"x": np.array(x), "y": np.array(y), "group": np.array(group), "size": np.array(size)}


def site_metrics(*, seed: int = 25, n_sites: int = 28) -> dict[str, Any]:
    """Per-site bias and spread for a caterpillar (forest) plot.

    Returns:
        Dict with ``labels`` (list of site names), ``bias``, ``std`` and ``lat``
        (``(n_sites,)`` arrays), ``layer`` (``"surface"`` or ``"free-trop"`` per
        site) and ``baseline``, a ``(bias, std)`` pair for a second series.
    """
    rng = _rng(seed)
    labels = [f"{c}{i:02d}" for i, c in zip(range(n_sites), rng.choice(list("ABCDEFG"), n_sites))]
    lat = rng.uniform(-80, 80, n_sites)
    layer = np.where(rng.uniform(0, 1, n_sites) < 0.6, "surface", "free-trop")
    bias = 0.3 * np.sin(np.deg2rad(lat)) + rng.normal(0, 0.35, n_sites) + (layer == "surface") * 0.1
    std = 0.4 + 0.5 * rng.uniform(0, 1, n_sites) + (layer == "surface") * 0.3
    base_bias = bias + rng.normal(0.15, 0.25, n_sites)
    base_std = std * rng.uniform(1.0, 1.4, n_sites)
    return {
        "labels": labels,
        "bias": bias,
        "std": std,
        "lat": lat,
        "layer": layer,
        "baseline": (base_bias, base_std),
    }


def cross_section_field(*, seed: int = 27, n_lat: int = 48, n_lev: int = 20) -> dict[str, np.ndarray]:
    """A latitude-pressure field and its error, for a cross-section.

    Returns:
        Dict with ``lats`` (``(n_lat,)``, °N), ``levels`` (``(n_lev,)``, hPa,
        descending from the surface) and ``value`` and ``error`` of shape
        ``(n_lat, n_lev)``.
    """
    rng = _rng(seed)
    lats = np.linspace(-88, 88, n_lat)
    levels = np.linspace(1000, 50, n_lev)  # hPa, descending (surface→top)
    la, lv = np.meshgrid(lats, levels, indexing="ij")
    value = (
        410
        + 6 * np.cos(np.deg2rad(la))
        - 0.01 * (1000 - lv)
        + 3 * np.exp(-((la - 40) ** 2) / 800) * (lv > 700)
    )
    error = 0.5 + 0.4 * rng.normal(0, 1, (n_lat, n_lev)) + 0.5 * np.exp(-((la) ** 2) / 500)
    return {"lats": lats, "levels": levels, "value": value, "error": error}


def global_grid(
    *, seed: int = 29, res: float = 2.0, field: str = "temperature"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A field on a regular global lon/lat grid.

    Args:
        seed: Random seed.
        res: Grid spacing in degrees.
        field: ``"temperature"`` (°C) or ``"anomaly"`` (zero-centred).

    Returns:
        ``(lon, lat, z)``: 1-D cell-centre ``lon`` and ``lat`` (north first) and
        ``z`` of shape ``(nlat, nlon)``.
    """
    rng = _rng(seed)
    lon = np.arange(-180 + res / 2, 180, res)
    lat = np.arange(90 - res / 2, -90, -res)
    lo, la = np.meshgrid(lon, lat)
    smooth = (
        np.sin(np.deg2rad(la) * 2) + 0.6 * np.cos(np.deg2rad(lo) * 1.5) + 0.4 * np.sin(np.deg2rad(lo + la))
    )
    if field == "anomaly":
        z = 2.2 * smooth + rng.normal(0, 0.3, lo.shape)
    else:
        z = 28 - 0.55 * np.abs(la) + 4 * smooth + rng.normal(0, 1.0, lo.shape)
    return lon, lat, z


def global_dataarray(*, seed: int = 29, res: float = 2.5, name: str = "t2m") -> xr.DataArray:
    """The :func:`global_grid` temperature field as an xarray ``DataArray``.

    Requires xarray. The dims are named ``lat`` and ``lon``, so
    :func:`nicefigs.map` accepts the array directly.
    """
    import xarray as xr

    lon, lat, z = global_grid(seed=seed, res=res)
    return xr.DataArray(z, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)


def icosahedral_field(
    *, seed: int = 31, n: int = 4000, field: str = "anomaly"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values on quasi-uniform unstructured global points (a Fibonacci sphere).

    The unstructured-mesh case for :func:`nicefigs.tri_map`.

    Args:
        seed: Random seed.
        n: Number of points.
        field: ``"anomaly"`` (zero-centred) or ``"temperature"`` (°C).

    Returns:
        ``(lon, lat, values)``, three ``(n,)`` arrays; degrees.
    """
    rng = _rng(seed)
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / n))
    lon = (np.degrees(np.pi * (1 + 5**0.5) * i) + 180) % 360 - 180
    smooth = (
        np.sin(np.deg2rad(lat) * 2)
        + 0.6 * np.cos(np.deg2rad(lon) * 1.5)
        + 0.5 * np.sin(np.deg2rad(lon + lat))
    )
    if field == "temperature":
        vals = 28 - 0.5 * np.abs(lat) + 4 * smooth + rng.normal(0, 1.0, n)
    else:
        vals = 2.2 * smooth + rng.normal(0, 0.25, n)
    return lon, lat, vals


def map_timeseries_cube(
    *, seed: int = 21, n_frames: int = 24, shape: tuple[int, int] = (90, 180)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A blob drifting around the globe, for animations.

    Returns:
        ``(lon, lat, frames)``: 2-D ``lon`` and ``lat`` of ``shape`` and
        ``frames`` of shape ``(n_frames, *shape)``.
    """
    rng = _rng(seed)
    ny, nx = shape
    lat = np.linspace(90, -90, ny)[:, None]
    lon = np.linspace(-180, 180, nx)[None, :]
    frames = np.empty((n_frames, ny, nx))
    for t in range(n_frames):
        cx = -180 + 360 * (t / n_frames)
        cy = 30 * np.sin(2 * np.pi * t / n_frames)
        base = np.exp(-(((lon - cx) / 40) ** 2 + ((lat - cy) / 25) ** 2))
        base += 0.3 * _land_weight(lon * np.ones_like(lat), lat * np.ones_like(lon))
        frames[t] = base + rng.normal(0, 0.02, (ny, nx))
    latg = np.repeat(lat, nx, axis=1)
    long = np.repeat(lon, ny, axis=0)
    return long, latg, frames


def taylor_stats(*, seed: int = 33, n: int = 6, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Per-model statistics for a Taylor diagram.

    Args:
        seed: Random seed.
        n: Number of default models; ignored when ``models`` is given.
        models: Model names.

    Returns:
        One dict per model with ``name``, ``correlation``, ``std_ratio`` (model
        over reference standard deviation) and ``crmsd`` (centred RMS
        difference, in reference standard deviations), consistent with each
        other.
    """
    rng = _rng(seed)
    models = (
        models or ["advection", "+ fluxes", "+ convection", "+ diffusion", "+ combined", "reanalysis"][:n]
    )
    out = []
    for i, name in enumerate(models):
        corr = float(np.clip(0.72 + 0.045 * i + rng.normal(0, 0.02), 0, 0.999))
        std_ratio = float(np.clip(0.7 + 0.09 * i + rng.normal(0, 0.05), 0.3, 1.6))
        crmsd = float(np.sqrt(max(1 + std_ratio**2 - 2 * std_ratio * corr, 0)))
        out.append({"name": name, "correlation": corr, "std_ratio": std_ratio, "crmsd": crmsd})
    return out


def hovmoller_field(*, seed: int = 35, n_time: int = 120, n_lat: int = 48) -> dict[str, Any]:
    """A time-latitude field with a seasonal cycle, and a reference version of it.

    ``value`` is ``reference`` plus a small structured bias and noise.

    Returns:
        Dict with ``times`` (``n_time`` ``datetime`` objects, 3 days apart from
        2020-01-01), ``lats`` (``(n_lat,)``, °N) and ``value`` and ``reference``
        of shape ``(n_time, n_lat)``.
    """
    rng = _rng(seed)
    lats = np.linspace(-88, 88, n_lat)
    t = np.arange(n_time)
    times = [_dt.datetime(2020, 1, 1) + _dt.timedelta(days=3 * int(k)) for k in t]
    season = np.sin(2 * np.pi * t[:, None] / (365 / 3))
    band = np.cos(np.deg2rad(lats))[None, :]
    reference = 410 + 4 * season * band + 3 * np.cos(np.deg2rad(lats))[None, :]
    value = (
        reference + 0.8 * season * np.sin(np.deg2rad(lats))[None, :] + rng.normal(0, 0.15, (n_time, n_lat))
    )
    return {"times": times, "lats": lats, "value": value, "reference": reference}


def vertical_profiles(
    *, seed: int = 37, n_lev: int = 20, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Named vertical profiles and a baseline, for :func:`nicefigs.profile`.

    Returns:
        Dict with ``levels`` (``(n_lev,)``, hPa, descending from the surface),
        ``profiles`` (``{name: (n_lev,) array}``) and ``baseline``
        (``(n_lev,)``).
    """
    rng = _rng(seed)
    names = names or ["model A", "model B", "model C"]
    levels = np.linspace(1000, 50, n_lev)
    shape = np.exp(-((levels - 1000) ** 2) / 6.0e5)
    profiles = {}
    for i, nm in enumerate(names):
        profiles[nm] = 405 + 8 * shape + 0.6 * i + rng.normal(0, 0.3, n_lev)
    baseline = 405 + 8 * shape
    return {"levels": levels, "profiles": profiles, "baseline": baseline}


def eigen_spectra(
    *,
    seed: int = 39,
    n_features: int = 256,
    n_samples: int = 192,
    names: Sequence[str] | None = None,
    alphas: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Named covariance eigenvalue spectra, for :func:`nicefigs.spectrum`.

    Each spectrum holds the eigenvalues, descending, of the sample covariance of
    ``n_samples`` draws whose feature variances decay as ``k^-alpha``. With
    ``n_samples < n_features`` the tail is numerically zero and may be slightly
    negative, as for a real rank-deficient sample covariance.

    Args:
        seed: Random seed.
        n_features: Number of features, the length of each spectrum.
        n_samples: Number of draws per covariance estimate.
        names: Series names; defaults to three forecast lead times.
        alphas: Power-law exponent of each series' feature variances.

    Returns:
        ``{name: eigenvalues}``, each of shape ``(n_features,)``.
    """
    rng = _rng(seed)
    names = list(names or ["lead 6 h", "lead 24 h", "lead 72 h"])
    alphas = list(alphas or [1.4, 1.1, 0.8][: len(names)])
    k = np.arange(1, n_features + 1)
    out = {}
    for nm, alpha in zip(names, alphas):
        x = rng.normal(size=(n_samples, n_features)) * k ** (-alpha / 2)
        x -= x.mean(axis=0)
        out[nm] = np.linalg.eigvalsh(x.T @ x / (n_samples - 1))[::-1]
    return out
