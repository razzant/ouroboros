# Ouroboros — common development commands
# Run `make` or `make help` to list the targets.

.DEFAULT_GOAL := help
.PHONY: help install run run-desktop test test-v test-web lint lint-web \
	check health docker-build docker-run sync-upstream clean

HOST ?= 127.0.0.1
PORT ?= 8765
UPSTREAM_REMOTE ?= managed
UPSTREAM_BRANCH ?= ouroboros
DOCKER_IMAGE ?= ouroboros-web

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## List the targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install: ## Install Python (uv, locked) and web (npm ci) dependencies
	uv sync --locked
	cd web && npm ci --no-audit --no-fund

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run: ## Run the HTTP server (HOST=127.0.0.1 PORT=8765 overridable)
	uv run --locked ouroboros server --host $(HOST) --port $(PORT)

run-desktop: ## Run the desktop launcher (native window, falls back to browser)
	uv run --locked python launcher.py

# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

test: ## Run Python tests (fast lane, no external deps at runtime)
	uv run --locked python -m pytest tests/ -q --tb=short

test-v: ## Run Python tests with verbose output
	uv run --locked python -m pytest tests/ -v --tb=long

test-web: ## Run browser-module tests (node --test, no framework)
	cd web && node --test tests/*.test.js

lint: ## Lint Python: deterministic F-rule gate, matches the CI quick-test step
	uv run --locked python -m ruff check . --select F

lint-web: ## Lint web modules: ESLint no-undef gate (needs `make install`)
	cd web && npm run lint:undef

check: lint lint-web test test-web ## Run every lint and test lane

health: ## Print codebase complexity metrics (requires ouroboros importable)
	uv run --locked python -c "from ouroboros.review import collect_sections, compute_complexity_metrics; \
		import pathlib, json; \
		sections, stats = collect_sections(pathlib.Path('.'), pathlib.Path('../data')); \
		m = compute_complexity_metrics(sections); \
		print(json.dumps({'repo': stats, **m}, indent=2, default=str))"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the server image ($(DOCKER_IMAGE))
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## Run the server image on PORT; set OUROBOROS_NETWORK_PASSWORD for non-local bind
	docker run --rm -p $(PORT):8765 \
	  -e OUROBOROS_NETWORK_PASSWORD="$$OUROBOROS_NETWORK_PASSWORD" \
	  $(DOCKER_IMAGE)

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

sync-upstream: ## Fetch upstream and merge $(UPSTREAM_REMOTE)/$(UPSTREAM_BRANCH) into the current branch
	git fetch $(UPSTREAM_REMOTE)
	git merge --no-edit $(UPSTREAM_REMOTE)/$(UPSTREAM_BRANCH)

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
