# Carbon Tracker Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help dev-setup build test clean deploy-local deploy-prod lint format security-scan

# Default target
help:
	@echo "Carbon Footprint Tracker - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev-setup     - Set up development environment"
	@echo "  run-local     - Run the application locally with Docker Compose"
	@echo "  run-dev       - Run in development mode with hot reload"
	@echo "  stop          - Stop all running services"
	@echo ""
	@echo "Building:"
	@echo "  build         - Build all Docker images"
	@echo "  build-api     - Build API service image"
	@echo "  build-producer - Build data producer image"
	@echo "  build-processor - Build stream processor image"
	@echo "  build-ml      - Build ML service image"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-performance - Run performance tests"
	@echo "  test-coverage - Generate test coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code"
	@echo "  type-check    - Run type checking"
	@echo "  security-scan - Run security vulnerability scan"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-local  - Deploy to local Kubernetes"
	@echo "  deploy-staging - Deploy to staging environment"
	@echo "  deploy-prod   - Deploy to production environment"
	@echo ""
	@echo "Database:"
	@echo "  db-init       - Initialize database schema"
	@echo "  db-migrate    - Run database migrations"
	@echo "  db-seed       - Seed database with sample data"
	@echo "  db-backup     - Create database backup"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs          - View application logs"
	@echo "  metrics       - View system metrics"
	@echo "  health-check  - Check system health"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         - Clean up temporary files and containers"
	@echo "  clean-all     - Clean everything including volumes"

# Variables
DOCKER_COMPOSE = docker-compose
KUBECTL = kubectl
PYTHON = python3
PIP = pip3
NODE = node
NPM = npm

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh
	@echo "Development environment ready!"

# Local development
run-local:
	@echo "Starting Carbon Tracker locally..."
	$(DOCKER_COMPOSE) up -d
	@echo "Services starting... Check http://localhost for dashboard"
	@echo "API available at http://localhost:8000"
	@echo "Grafana available at http://localhost:3000"

run-dev:
	@echo "Starting in development mode..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up
	
stop:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

# Building
build:
	@echo "Building all Docker images..."
	$(DOCKER_COMPOSE) build

build-api:
	@echo "Building API service..."
	docker build -f Dockerfile.api -t carbon-tracker-api .

build-producer:
	@echo "Building data producer..."
	docker build -f Dockerfile.producer -t carbon-tracker-producer .

build-processor:
	@echo "Building stream processor..."
	docker build -f Dockerfile.processor -t carbon-tracker-processor .

build-ml:
	@echo "Building ML service..."
	docker build -f Dockerfile.ml -t carbon-tracker-ml .

# Testing
test: test-unit test-integration
	@echo "All tests completed!"

test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

test-integration:
	@echo "Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v

test-performance:
	@echo "Running performance tests..."
	@if command -v k6 >/dev/null 2>&1; then \
		k6 run tests/performance/load_test.js; \
	else \
		echo "k6 not installed. Install from https://k6.io/docs/getting-started/installation/"; \
	fi

test-coverage:
	@echo "Generating test coverage report..."
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/"

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "Code formatted!"

type-check:
	@echo "Running type checks..."
	mypy src/ --ignore-missing-imports

security-scan:
	@echo "Running security vulnerability scan..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r src/; \
	else \
		echo "Installing bandit..."; \
		$(PIP) install bandit; \
		bandit -r src/; \
	fi
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "Installing safety..."; \
		$(PIP) install safety; \
		safety check; \
	fi

# Database operations
db-init:
	@echo "Initializing database..."
	$(DOCKER_COMPOSE) exec postgres psql -U carbon_user -d carbon_tracker -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	@echo "Running database migrations..."
	$(PYTHON) scripts/migrate_db.py

db-seed:
	@echo "Seeding database with sample data..."
	$(PYTHON) scripts/seed_data.py

db-backup:
	@echo "Creating database backup..."
	$(DOCKER_COMPOSE) exec postgres pg_dump -U carbon_user carbon_tracker > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created!"

# Deployment
deploy-local:
	@echo "Deploying to local Kubernetes..."
	$(KUBECTL) apply -f k8s/namespace.yaml
	$(KUBECTL) apply -f k8s/configmaps.yaml
	$(KUBECTL) apply -f k8s/secrets.yaml
	$(KUBECTL) apply -f k8s/
	@echo "Deployed to local Kubernetes!"

deploy-staging:
	@echo "Deploying to staging environment..."
	@if [ -z "$(KUBE_CONFIG_STAGING)" ]; then \
		echo "KUBE_CONFIG_STAGING environment variable not set"; \
		exit 1; \
	fi
	$(KUBECTL) --kubeconfig=$(KUBE_CONFIG_STAGING) apply -f k8s/ -n carbon-tracker-staging
	@echo "Deployed to staging!"

deploy-prod:
	@echo "Deploying to production environment..."
	@if [ -z "$(KUBE_CONFIG_PRODUCTION)" ]; then \
		echo "KUBE_CONFIG_PRODUCTION environment variable not set"; \
		exit 1; \
	fi
	@read -p "Are you sure you want to deploy to production? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(KUBECTL) --kubeconfig=$(KUBE_CONFIG_PRODUCTION) apply -f k8s/ -n carbon-tracker
	@echo "Deployed to production!"

# Infrastructure
terraform-init:
	@echo "Initializing Terraform..."
	cd terraform && terraform init

terraform-plan:
	@echo "Planning Terraform changes..."
	cd terraform && terraform plan

terraform-apply:
	@echo "Applying Terraform changes..."
	cd terraform && terraform apply

terraform-destroy:
	@echo "Destroying Terraform infrastructure..."
	@read -p "Are you sure you want to destroy infrastructure? [y/N] " confirm && [ "$$confirm" = "y" ]
	cd terraform && terraform destroy

# Monitoring
logs:
	@echo "Viewing application logs..."
	$(DOCKER_COMPOSE) logs -f

logs-api:
	@echo "Viewing API logs..."
	$(DOCKER_COMPOSE) logs -f carbon-api

logs-producer:
	@echo "Viewing producer logs..."
	$(DOCKER_COMPOSE) logs -f carbon-producer

logs-processor:
	@echo "Viewing processor logs..."
	$(DOCKER_COMPOSE) logs -f carbon-processor

metrics:
	@echo "Opening metrics dashboard..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

health-check:
	@echo "Checking system health..."
	@curl -f http://localhost:8000/health || echo "API health check failed"
	@curl -f http://localhost:3000/api/health || echo "Grafana health check failed"
	@curl -f http://localhost:9090/-/healthy || echo "Prometheus health check failed"

# Frontend development
frontend-install:
	@echo "Installing frontend dependencies..."
	cd frontend && $(NPM) install

frontend-dev:
	@echo "Starting frontend development server..."
	cd frontend && $(NPM) run dev

frontend-build:
	@echo "Building frontend for production..."
	cd frontend && $(NPM) run build

frontend-test:
	@echo "Running frontend tests..."
	cd frontend && $(NPM) test

# Cleanup
clean:
	@echo "Cleaning up temporary files and containers..."
	$(DOCKER_COMPOSE) down
	docker system prune -f
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -delete
	@rm -rf htmlcov/
	@rm -rf .coverage
	@echo "Cleanup completed!"

clean-all: clean
	@echo "Cleaning everything including volumes..."
	$(DOCKER_COMPOSE) down -v
	docker system prune -a -f --volumes
	@echo "Deep cleanup completed!"

# Development utilities
shell-api:
	@echo "Opening shell in API container..."
	$(DOCKER_COMPOSE) exec carbon-api /bin/bash

shell-db:
	@echo "Opening database shell..."
	$(DOCKER_COMPOSE) exec postgres psql -U carbon_user -d carbon_tracker

shell-redis:
	@echo "Opening Redis shell..."
	$(DOCKER_COMPOSE) exec redis redis-cli

# Documentation
docs-build:
	@echo "Building documentation..."
	@if [ -d "docs" ]; then \
		cd docs && make html; \
	else \
		echo "Documentation directory not found"; \
	fi

docs-serve:
	@echo "Serving documentation..."
	@if [ -d "docs/_build/html" ]; then \
		cd docs/_build/html && $(PYTHON) -m http.server 8080; \
	else \
		echo "Documentation not built. Run 'make docs-build' first"; \
	fi

# Quick start
quick-start: dev-setup build run-local
	@echo ""
	@echo "🚀 Carbon Tracker is now running!"
	@echo ""
	@echo "📊 Dashboard: http://localhost"
	@echo "🔧 API: http://localhost:8000"
	@echo "📈 Grafana: http://localhost:3000 (admin/carbontracker123)"
	@echo "📊 Prometheus: http://localhost:9090"
	@echo ""
	@echo "To stop: make stop"
	@echo "To view logs: make logs"
	@echo "To run tests: make test"