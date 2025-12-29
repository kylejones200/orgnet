.PHONY: format lint test clean

format:
	black onapy/ tests/ example.py setup.py --line-length 100

lint:
	flake8 onapy/ tests/ example.py setup.py --max-line-length=100 --ignore=E203,W503,E501

test:
	pytest tests/ -v

check: format lint test

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true

