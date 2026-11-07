# Development tasks, all run through uv.

.PHONY: dev lint test check docs docs-check docs-figures serve-docs gallery logo build

dev:
	uv pip install -e ".[dev,docs]"
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

test:
	MPLBACKEND=Agg uv run pytest

# What CI enforces: pre-commit (ruff lint + format, file hygiene) and the test suite.
check: lint test

docs:
	uv run mkdocs build --strict

# Run every README/docs code example and compare it with the output shown.
docs-check:
	MPLBACKEND=Agg uv run python scripts/docs_examples.py

# Re-render the guide figures in docs/assets/guide/ from the guide's code examples.
docs-figures:
	MPLBACKEND=Agg uv run python scripts/docs_examples.py --render

serve-docs:
	uv run mkdocs serve

# Re-render every example into docs/gallery/ and the README images into docs/assets/.
gallery:
	MPLBACKEND=Agg uv run python examples/render_gallery.py

# Re-draw the logo files in docs/assets/logo/.
logo:
	uv run python scripts/make_logo.py

build:
	rm -rf dist
	uv run python -m build
	uv run twine check --strict dist/*
