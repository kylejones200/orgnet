.PHONY: format lint test clean pre-push install-hooks check build

format:
	black orgnet/ tests/ example.py setup.py --line-length 100

lint:
	flake8 orgnet/ tests/ example.py setup.py --max-line-length=100 --ignore=E203,W503,E501

test:
	pytest tests/ -v

build:
	python -m build
	twine check dist/*

pre-push:
	@echo "Running pre-push checks..."
	@./scripts/pre-push.sh

install-hooks:
	@./scripts/install-git-hooks.sh

check: format lint test build

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

