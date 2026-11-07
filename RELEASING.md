# Releasing

Releases are cut from `main` by pushing a `vX.Y.Z` tag; the `release` workflow does the rest.

1. **Bump the version** in `src/nicefigs/__init__.py` (`__version__ = "X.Y.Z"`, following
   [semantic versioning](https://semver.org)). `pyproject.toml` reads it from there.
2. **Update `CHANGELOG.md`**: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and open a fresh
   `## [Unreleased]` above it. The release notes are taken verbatim from this section.
3. **Merge** both changes to `main` through a pull request, with CI green.
4. **Tag and push** the merge commit:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

The workflow checks the tag matches `__version__`, builds the sdist and wheel, checks them with
`twine`, runs the test suite against the installed wheel, publishes to PyPI via trusted publishing,
and creates the GitHub release with the changelog section and the distributions attached.

## One-time setup

- On PyPI, add a trusted publisher for project `nicefigs`: owner `EarthyScience`, repository
  `nicefigs`, workflow `release.yml`, environment `pypi`.
- On GitHub, create the `pypi` environment (Settings → Environments); restrict it to `v*` tags.
- For the docs site, set Pages to deploy from the `gh-pages` branch.
