# EDCOCR Makefile
#
# Convenience targets for development, testing, and deployment.
# Run `make help` to list available targets.

.PHONY: help install install-dev test test-fast lint format build run stop logs clean smoke \
        docker-build docker-up docker-down docker-logs \
        helm-lint helm-template helm-install \
        sdk-build-python sdk-build-typescript \
        docs-validate presentation-open

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Python environment
# ============================================================================

install: ## Install runtime Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install ## Install development dependencies + pre-commit
	pip install pre-commit ruff pytest pytest-cov
	pre-commit install

# ============================================================================
# Testing
# ============================================================================

test: ## Run the full test suite
	python -m pytest tests/ -v

test-fast: ## Run tests excluding slow integration tests
	python -m pytest tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

smoke: ## Run the smoke pipeline against the sample fixture
	python scripts/smoke_pipeline.py

# ============================================================================
# Code quality
# ============================================================================

lint: ## Run ruff linter
	ruff check .

format: ## Auto-format with ruff
	ruff format .
	ruff check --fix .

# ============================================================================
# Running locally (bare metal)
# ============================================================================

run: ## Run the production async pipeline
	python ocr_gpu_async.py

run-api: ## Run the FastAPI server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-coordinator: ## Run the Django coordinator
	cd coordinator && python manage.py runserver 0.0.0.0:8001

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start the Docker stack
	docker compose up -d

docker-down: ## Stop the Docker stack
	docker compose down

docker-logs: ## Tail Docker logs
	docker compose logs -f

docker-restart: docker-down docker-up ## Restart the Docker stack

docker-clean: ## Stop and remove all Docker artifacts
	docker compose down -v --remove-orphans
	docker system prune -af

# CPU-only variant
docker-cpu-up: ## Start the CPU-only Docker stack (no GPU required)
	docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d --build

# ============================================================================
# Kubernetes / Helm
# ============================================================================

helm-lint: ## Lint the Helm chart
	helm lint helm/ocr-local/

helm-template: ## Render Helm chart templates to stdout
	helm template edcocr helm/ocr-local/ -f helm/ocr-local/values.yaml

helm-install: ## Install the Helm chart (requires values-secret.yaml)
	@test -f values-secret.yaml || (echo "ERROR: values-secret.yaml not found. Create it with required secrets." && exit 1)
	helm install edcocr helm/ocr-local/ \
		--namespace edcocr \
		--create-namespace \
		-f helm/ocr-local/values.yaml \
		-f values-secret.yaml

helm-upgrade: ## Upgrade an existing Helm release
	helm upgrade edcocr helm/ocr-local/ \
		--namespace edcocr \
		-f helm/ocr-local/values.yaml \
		-f values-secret.yaml

helm-uninstall: ## Uninstall the Helm release
	helm uninstall edcocr --namespace edcocr

# ============================================================================
# SDK builds
# ============================================================================

sdk-build-python: ## Build the Python SDK distribution
	cd sdk/python && python -m build

sdk-build-typescript: ## Build the TypeScript SDK
	cd sdk/typescript && npm install && npm run build

sdk-test-python: ## Test the Python SDK
	cd sdk/python && python -m pytest tests/ -v

sdk-test-typescript: ## Test the TypeScript SDK
	cd sdk/typescript && npm test

# ============================================================================
# Documentation
# ============================================================================

docs-validate: ## Validate documentation links and references
	@if [ -f scripts/check_docs.py ]; then python scripts/check_docs.py; else echo "scripts/check_docs.py not present"; fi

presentation-open: ## Open the HTML presentation in a browser
	@python -c "import webbrowser; webbrowser.open('presentation/index.html')"

# ============================================================================
# Models
# ============================================================================

download-models: ## Pre-download PaddleOCR language models
	python download_models.py

download-models-cpu: ## Pre-download models for CPU-only environments
	python download_models.py --cpu-only

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Remove caches and temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage build/ dist/ *.egg-info/

clean-output: ## Remove all OCR outputs and temp files
	rm -rf ocr_output/EXPORT ocr_output/logs ocr_temp/

# ============================================================================
# Release helpers
# ============================================================================

version: ## Show current EDCOCR version
	@python -c "from version import __version__; print(__version__)"

bump-patch: ## Bump patch version (X.Y.Z+1)
	@python scripts/bump_version.py patch

bump-minor: ## Bump minor version (X.Y+1.0)
	@python scripts/bump_version.py minor

bump-major: ## Bump major version (X+1.0.0)
	@python scripts/bump_version.py major
