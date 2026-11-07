"""Render every example into ``docs/gallery/`` (PNG) and the README images into ``docs/assets/``.

Each ``NN_*.py`` script exposes ``make()``, which returns a Figure or, for an
animation, writes the file it is given. Run from the repo root::

    python examples/render_gallery.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent
GALLERY = ROOT / "docs" / "gallery"
ASSETS = ROOT / "docs" / "assets"

#: Resolution of every rendered PNG; one image pixel per CSS pixel makes a 5.5 in figure 1100 px wide.
DPI = 200

#: Examples also written as PNG into ``docs/assets/`` for the README.
README_FIGURES = (
    "01_global_map",
    "05_ternary_map",
    "14_agreement",
    "17_model_comparison",
    "20_scorecard",
    "23_composite_figure",
)


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
    scripts += sorted(p for p in HERE.glob("guide_*.py"))
    results = {}
    for script in scripts:
        name = script.stem
        t0 = time.time()
        try:
            mod = _load(script)
            if "animation" in name:
                mod.make(str(GALLERY / f"{name}.gif"))
            else:
                fig = mod.make()
                fig.savefig(
                    GALLERY / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white"
                )
                if name in README_FIGURES:
                    fig.savefig(
                        ASSETS / f"{name}.png",
                        dpi=DPI,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        facecolor="white",
                    )
                plt.close(fig)
            results[name] = f"OK ({time.time() - t0:.1f}s)"
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            results[name] = f"FAIL: {e}"
    print("\n===== GALLERY =====")
    for k, v in results.items():
        print(f"  {k:26s} {v}")
    n_fail = sum(1 for v in results.values() if not v.startswith("OK"))
    print(f"\n{len(results) - n_fail}/{len(results)} rendered")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
