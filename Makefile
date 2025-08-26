.PHONY: help install test benchmark clean lint format

help:
	@echo "MiniLLM Project Management"
	@echo "========================="
	@echo "Available commands:"
	@echo "  install     - Install the package in development mode"
	@echo "  test        - Run all tests"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean build artifacts"
	@echo "  demo        - Run usage demo"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	@echo "Running integration tests..."
	python tests/test_integration.py
	@echo "Running KV Cache tests..."
	python tests/test_kvcache.py

benchmark:
	@echo "Running performance analysis..."
	python benchmarks/performance_analysis.py

demo:
	@echo "Running usage demo..."
	python examples/demo_usage.py

lint:
	flake8 src/ tests/ benchmarks/ examples/

format:
	black src/ tests/ benchmarks/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick development workflow
dev-setup: install-dev
	@echo "Development environment ready!"

# Full test suite
test-all: test benchmark demo
	@echo "All tests completed!"

# Pre-commit checks
check: lint test
	@echo "Pre-commit checks passed!"
