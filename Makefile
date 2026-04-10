.PHONY: install dev install-docs examples fix lint type-check test docs docs-serve docs-deploy check clean help

help:
	@echo "Dev (modify files):  fix"
	@echo "Checks (read-only):  lint | type-check | test | docs | check"
	@echo "Setup:               install | dev | install-docs"
	@echo "Docs:                docs-serve | docs-deploy"
	@echo "Cleanup:             clean"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync

dev:
	uv sync --extra dev

install-docs:
	uv sync --extra docs

examples:
	uv sync --extra examples

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	uv run ruff format .
	uv run ruff check --fix .

# ── Checks (read-only — mirrors GitHub CI) ─────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .

type-check:
	uv run ty check linmult

test:
	uv run coverage run -m unittest discover -s tests -v
	uv run coverage report
	uv run coverage html
	uv run coverage xml -o coverage.xml

docs:
	uv run sphinx-build -b html docs/ site/

check: lint type-check test docs

# ── Docs ───────────────────────────────────────────────────────────────────────

docs-serve:
	uv run sphinx-autobuild docs/ site/

docs-deploy:
	@echo "Docs are deployed automatically via GitHub Actions on push to main."

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf .venv coverage_html dist/ .pytest_cache/ site/ tmp/ examples/data/
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
