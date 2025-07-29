.PHONY: install test lint format clean train serve mlflow-ui

# Variables
PYTHON = python
PIP = pip
UV = uv
PROJECT_NAME = implicit_vibecode
TEST_PATH = tests/

# Install dependencies
install:
	$(UV) pip install -r requirements/base.txt
	$(UV) pip install -r requirements/dev.txt

# Install in development mode
dev: install
	$(UV) pip install -e .

# Run tests
test:
	pytest $(TEST_PATH) -v

# Run linting
lint:
	black --check $(PROJECT_NAME) $(TEST_PATH)
	flake8 $(PROJECT_NAME) $(TEST_PATH)
	mypy $(PROJECT_NAME) $(TEST_PATH)

# Format code
format:
	black $(PROJECT_NAME) $(TEST_PATH)
	isort $(PROJECT_NAME) $(TEST_PATH)

# Clean up
clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf .coverage htmlcov
	rm -rf mlruns
	rm -rf .pytest_cache

# Train model
train:
	$(PYTHON) -m $(PROJECT_NAME).training.trainer

# Start MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri=./mlruns

# Run the API server
serve:
	uvicorn $(PROJECT_NAME).api.app:app --reload
