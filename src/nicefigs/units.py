"""Axis labels of the form ``Quantity [unit]``, unit prettifying, and number formatting."""

from __future__ import annotations

import re

import numpy as np
from matplotlib.axes import Axes

_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

_UNIT_FIX = {
    "degC": "°C",
    "deg": "°",
    "degrees": "°",
    "um": "µm",
    "ug": "µg",
    "percent": "%",
}


def prettify_unit(unit: str | None, *, superscript: bool = True) -> str:
    """Prettify a plain-text unit string.

    ``"m2"`` becomes ``"m²"``, ``"kg m-2 s-1"`` becomes ``"kg m⁻² s⁻¹"`` and
    ``"degC"`` becomes ``"°C"``.

    Args:
        unit: Plain-text unit; ``None`` gives ``""``.
        superscript: Render exponents as Unicode superscripts; otherwise as ``^n``.

    Returns:
        The prettified unit.
    """
    if unit is None:
        return ""
    u = unit.strip()
    if u in _UNIT_FIX:
        return _UNIT_FIX[u]
    for k, v in _UNIT_FIX.items():
        u = re.sub(rf"\b{k}\b", v, u)

    def _exp(m: re.Match) -> str:
        base, exp = m.group(1), m.group(2)
        if superscript:
            return base + exp.translate(_SUP)
        return f"{base}^{exp}"

    u = re.sub(r"([A-Za-zµ°%])(-?\d+)", _exp, u)
    u = re.sub(r"([A-Za-zµ°%])\^(-?\d+)", _exp, u)
    return u


#: Display names of common metric keys, used by :func:`metric_label`. Extend in place.
METRIC_NAMES: dict[str, str] = {
    "rmse": "RMSE",
    "mse": "MSE",
    "mae": "MAE",
    "bias": "Bias",
    "r2": "R²",
    "r": "r",
    "corr": "Correlation",
    "cosine": "Cosine similarity",
    "fve": "FVE",
    "nse": "NSE",
    "kge": "KGE",
    "auc": "AUC",
    "acc": "Accuracy",
    "f1": "F1",
    "l0": "L0 norm",
    "sparsity": "Sparsity",
    "morans_i": "Moran's I",
    "smoothness": "Smoothness",
    "skill": "Skill",
}

#: Direction (one of :data:`DIRECTIONS`) per metric key, for the arrows of
#: :func:`metric_label`. Extend in place.
METRIC_DIRECTION: dict[str, str] = {
    "rmse": "lower_better",
    "mse": "lower_better",
    "mae": "lower_better",
    "l0": "lower_better",
    "r2": "higher_better",
    "cosine": "higher_better",
    "fve": "higher_better",
    "nse": "higher_better",
    "kge": "higher_better",
    "auc": "higher_better",
    "acc": "higher_better",
    "f1": "higher_better",
    "skill": "higher_better",
}


#: The ways a metric can be better: larger, smaller, or closer to a target value.
DIRECTIONS: tuple[str, ...] = ("higher_better", "lower_better", "target")


def check_direction(direction: str) -> str:
    """Return ``direction`` if it is one of :data:`DIRECTIONS`.

    Raises:
        ValueError: ``direction`` is not in :data:`DIRECTIONS`.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; known: {list(DIRECTIONS)}")
    return direction


def _arrow(direction: str | None) -> str:
    if direction is None:
        return ""
    return {"higher_better": "↑", "lower_better": "↓", "target": ""}[check_direction(direction)]


def axis_label(
    quantity: str,
    units: str | None = None,
    *,
    superscript: bool = True,
    brackets: str = "[]",
    direction: str | None = None,
) -> str:
    """Compose an axis label ``"Quantity [unit]"``.

    Args:
        quantity: Quantity name.
        units: Plain-text unit, prettified with :func:`prettify_unit`; empty or
            ``None`` gives just the quantity.
        superscript: Render unit exponents as superscripts.
        brackets: Opening and closing bracket characters.
        direction: One of :data:`DIRECTIONS`; ``"higher_better"`` and
            ``"lower_better"`` append an arrow showing which way is better.

    Returns:
        The label.

    Raises:
        ValueError: ``direction`` is not in :data:`DIRECTIONS`.
    """
    label = quantity
    if units:
        lo, hi = brackets[0], brackets[1] if len(brackets) > 1 else brackets[0]
        label = f"{quantity} {lo}{prettify_unit(units, superscript=superscript)}{hi}"
    arrow = _arrow(direction)
    return f"{label} {arrow}".rstrip() if arrow else label


def metric_label(
    key: str,
    units: str | None = None,
    *,
    direction: str | bool | None = None,
    registry: dict[str, str] | None = None,
) -> str:
    """Axis label for a metric key, e.g. ``"rmse"`` gives ``"RMSE"``.

    Args:
        key: Metric key, looked up case-insensitively; unknown keys are title-cased.
        units: Plain-text unit.
        direction: ``True`` looks the arrow up in :data:`METRIC_DIRECTION`; a
            string sets it as in :func:`axis_label`; ``None`` or ``False`` omits it.
        registry: Key-to-name mapping used instead of :data:`METRIC_NAMES`.

    Returns:
        The label.
    """
    reg = registry or METRIC_NAMES
    name = reg.get(key.lower(), key.replace("_", " ").title())
    if direction is True:
        direction = METRIC_DIRECTION.get(key.lower())
    elif direction is False:
        direction = None
    return axis_label(name, units, direction=direction)


def set_labels(
    ax: Axes,
    x: str | None = None,
    y: str | None = None,
    *,
    xunits: str | None = None,
    yunits: str | None = None,
    title: str | None = None,
) -> None:
    """Set x and y labels (as :func:`axis_label`) and the title; ``None`` leaves one unchanged."""
    if x is not None:
        ax.set_xlabel(axis_label(x, xunits))
    if y is not None:
        ax.set_ylabel(axis_label(y, yunits))
    if title is not None:
        ax.set_title(title)


def fmt_val(v: float, *, sig: int = 3) -> str:
    """Format a number with ``sig`` significant digits.

    Magnitudes outside ``[1e-3, 1e4)`` use ``m×10ⁿ`` notation; zero gives ``"0"``
    and non-finite values give ``"–"``.
    """
    if v == 0 or not np.isfinite(v):
        return "0" if v == 0 else "–"
    a = abs(v)
    if 1e-3 <= a < 1e4:
        s = f"{v:.{sig}g}"
        return s
    exp = int(np.floor(np.log10(a)))
    mant = v / 10**exp
    return f"{mant:.{max(sig - 1, 0)}f}×10{str(exp).translate(_SUP)}"


def fmt_pval(p: float) -> str:
    """Format a p-value as ``"p < 0.001"``, ``"p = 0.004"`` or ``"p = 0.12"``."""
    if p < 1e-3:
        return "p < 0.001"
    if p < 1e-2:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"
