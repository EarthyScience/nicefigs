"""Run every code example in the docs and check its shown output.

Each ```` ```python ```` block in the README and the docs pages runs on its own,
in a fresh namespace and a scratch working directory, with the matplotlib state
of a fresh ``import nicefigs``. Directly below a block, the page may show

- its output, as a ```` ```text ```` block that must equal what the block prints;
- the figure it draws, as an image ``![...](path)``.

A block that prints must show a ```` ```text ```` block, and on the docs pages a
block that draws a figure must show an image. Every ``examples/NN_*.py`` script
is run too, through its ``make()``.

Usage, from the repository root::

    python scripts/docs_examples.py            # check
    python scripts/docs_examples.py --render   # also write images under docs/assets/guide/
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import nicefigs as nf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "README.md",
    "docs/index.md",
    "docs/quickstart.md",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs/guide").glob("*.md")),
]
RENDERED = ROOT / "docs" / "assets" / "guide"
RENDER_DPI = 200

_FENCE = re.compile(r"^```(\w*)\s*$")
_IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)\s]+)\)")


@dataclass
class Example:
    page: Path
    line: int
    code: str
    text: str | None
    image: Path | None


def _blocks(lines: list[str], i: int) -> tuple[str, str, int]:
    """Return ``(lang, body, next_line)`` for the fenced block opening at ``i``."""
    lang = _FENCE.match(lines[i]).group(1)
    j = i + 1
    while not lines[j].startswith("```"):
        j += 1
    return lang, "\n".join(lines[i + 1 : j]) + "\n", j + 1


def parse(page: Path) -> list[Example]:
    """The python examples of ``page``, each with the output shown below it."""
    lines = page.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        if not _FENCE.match(lines[i]):
            i += 1
            continue
        lang, body, nxt = _blocks(lines, i)
        if lang != "python" or body.lstrip().startswith("--8<--"):
            i = nxt
            continue
        text = image = None
        k = nxt
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k < len(lines) and _FENCE.match(lines[k]) and _FENCE.match(lines[k]).group(1) == "text":
            _, text, k = _blocks(lines, k)
            while k < len(lines) and not lines[k].strip():
                k += 1
        if k < len(lines) and (m := _IMAGE.match(lines[k])):
            image = (page.parent / m.group(1)).resolve()
        out.append(Example(page, i + 1, body, text, image))
        i = nxt
    return out


@contextlib.contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _fresh_state() -> None:
    plt.close("all")
    nf.set_theme("paper")
    mpl.rcdefaults()
    matplotlib.use("Agg")


def _run(code: str, name: str, workdir: Path) -> str:
    """Execute ``code`` as module ``name`` in ``workdir``; return its stdout."""
    module = types.ModuleType(name)
    sys.modules[name] = module  # lets worker processes unpickle functions defined in the block
    buf = io.StringIO()
    with _cwd(workdir), contextlib.redirect_stdout(buf):
        exec(compile(code, name, "exec"), module.__dict__)
    return buf.getvalue()


def check_page(page: Path, render: bool, workdir: Path) -> list[str]:
    """Run the examples of ``page``; return one message per failure."""
    errors = []
    strict = page.name != "README.md"
    rel = page.relative_to(ROOT)
    for n, ex in enumerate(parse(page)):
        where = f"{rel}:{ex.line}"
        _fresh_state()
        try:
            stdout = _run(ex.code, f"docblock_{page.stem}_{n}", workdir)
        except Exception:  # noqa: BLE001
            errors.append(f"{where}: raised\n{traceback.format_exc(limit=-3)}")
            continue
        shown = "\n".join(line.rstrip() for line in (ex.text or "").strip().splitlines())
        printed = "\n".join(line.rstrip() for line in stdout.strip().splitlines())
        if ex.text is not None and printed != shown:
            errors.append(
                f"{where}: output differs from the text block\n--- shown\n{shown}\n--- printed\n{printed}"
            )
        elif ex.text is None and printed:
            errors.append(f"{where}: prints output but shows no text block\n{printed}")
        figs = plt.get_fignums()
        if ex.image is None:
            if figs and strict:
                errors.append(f"{where}: draws a figure but shows no image")
            continue
        if ex.image.is_relative_to(RENDERED):
            if not figs:
                errors.append(f"{where}: shows {ex.image.name} but draws no figure")
            elif render:
                ex.image.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figs[-1]).savefig(ex.image, dpi=RENDER_DPI, bbox_inches="tight", facecolor="white")
        if not ex.image.exists():
            errors.append(f"{where}: image {ex.image.relative_to(ROOT)} does not exist")
    return errors


def check_examples(workdir: Path) -> list[str]:
    """Run ``make()`` of every gallery example script."""
    errors = []
    for script in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        _fresh_state()
        spec = importlib.util.spec_from_file_location(script.stem, script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[script.stem] = module
        try:
            with _cwd(workdir):
                spec.loader.exec_module(module)
                if "animation" in script.stem:
                    module.make(str(workdir / f"{script.stem}.gif"))
                else:
                    module.make()
        except Exception:  # noqa: BLE001
            errors.append(f"{script.relative_to(ROOT)}: raised\n{traceback.format_exc(limit=-3)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true", help="write the images under docs/assets/guide/")
    parser.add_argument("pages", nargs="*", help="pages to check (default: README and all docs pages)")
    args = parser.parse_args()
    pages = [ROOT / p for p in (args.pages or PAGES)]
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for page in pages:
            errors += check_page(page.resolve(), args.render, Path(tmp))
        if not args.pages:
            errors += check_examples(Path(tmp))
    _fresh_state()
    for e in errors:
        print(e, file=sys.stderr)
    print(f"{len(errors)} problem(s) in {len(pages)} page(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
