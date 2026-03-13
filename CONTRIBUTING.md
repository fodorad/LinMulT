# Contributing to LinMulT

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Quick start

```bash
git clone https://github.com/fodorad/linmult
cd linmult
pip install -e ".[dev,docs]"
pre-commit install   # optional: runs ruff automatically before every commit
```

To also run the benchmark example scripts (`examples/benchmark_urfunny.py`, etc.):

```bash
pip install gdown
```

---

## Development workflow

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep them focused and minimal.
3. **Write or update tests** in `tests/` to cover your changes.
4. **Run checks locally** before pushing:

   ```bash
   make fix    # auto-format and fix lint issues
   make check  # lint + tests + docs build (mirrors CI)
   ```

5. **Open a Pull Request** against `main` and fill in the template.

---

## Commit message convention

LinMulT follows **Conventional Commits** to make the version history readable
and to signal the correct version bump when a release is made.

| Prefix | Meaning | Version bump |
|--------|---------|--------------|
| `fix:` | Bug fix, regression, hotfix | **Patch** (2.0.x) |
| `feat:` | New feature, new config option | **Minor** (2.x.0) |
| `feat!:` or `BREAKING CHANGE:` | API or config change that breaks existing usage | **Major** (x.0.0) |
| `docs:` | Documentation only | No bump |
| `test:` | Tests only | No bump |
| `refactor:` | Code refactor with no behaviour change | No bump |
| `chore:` | Build, CI, dependency updates | No bump |

### Examples

```
fix: prevent NaN in attention pooling for empty sequences
feat: add performer (FAVOR+) attention type
feat!: rename head type 'agg' to 'sequence_aggregation'
docs: add getting-started examples for LinT
chore: bump ruff to v0.9
```

For a breaking change using a footer instead of `!`:

```
feat: overhaul config schema

BREAKING CHANGE: `num_heads` renamed to `n_heads` everywhere.
```

---

## Release process

Releases are **tag-driven**. The version is derived from the git tag at build
time (`hatch-vcs`) — there is no version number stored in `pyproject.toml` and
no "bump version" commit to make.

When you are ready to release:

```bash
# 1. Update CHANGELOG.md — rename [Unreleased] → [2.1.0] and add the date
# 2. Commit the changelog
git add CHANGELOG.md
git commit -m "chore: release v2.1.0"

# 3. Tag and push
git tag v2.1.0
git push origin main --tags
```

GitHub Actions then builds the wheel (version=2.1.0 from the tag), publishes to
PyPI, deploys docs, and creates a GitHub Release automatically.

**Version bump decision** — look at the commits since the last tag:

- Any `feat!` or `BREAKING CHANGE` → **major**
- Any `feat:` → **minor**
- Only `fix:`, `docs:`, `chore:` → **patch**

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) — run `make fix` to auto-apply.
- **Line length**: 100 characters.
- **Python version**: 3.12+.
- **Type hints**: encouraged but not required for internal helpers.

---

## Tests

```bash
make test              # run all tests with coverage
coverage html          # open coverage_html/index.html to browse
```

Test files live in `tests/` and mirror the `linmult/` package structure.
New attention variants need unit tests in `tests/core/test_attention.py`
and integration tests in `tests/models/test_linmult.py`.

---

## Reporting bugs and requesting features

Please use the GitHub issue templates:

- **Bug report**: include a minimal reproducible example, Python/PyTorch version, and OS.
- **Feature request**: describe the problem you are trying to solve, not just the solution.

---

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE).
