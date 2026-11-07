"""`model_comparison`: an unknown ``kind`` is an error, not an empty figure."""

from __future__ import annotations

import pytest

import nicefigs as nf


def test_model_comparison_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        nf.model_comparison({"m": {"a": [1.0, 2.0]}}, kind="bars")
