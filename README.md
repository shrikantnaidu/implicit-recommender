# Implicit Vibecode

A production-ready recommender system using implicit feedback with ALS (Alternating Least Squares) for collaborative filtering, featuring MLflow integration for experiment tracking and model serving.

## Features

- 🚀 **Implicit Feedback Recommendations**: Leverages user-item interaction data for personalized recommendations
- 📊 **MLflow Integration**: Track experiments, parameters, and metrics
- 🧪 **Model Evaluation**: Comprehensive metrics including NDCG and Precision@K
- 🚀 **Production-Ready API**: FastAPI-based serving with Swagger documentation
- 🔄 **Reproducible Training**: Deterministic training with configurable hyperparameters
- 📦 **Modular Design**: Clean, maintainable codebase with separation of concerns

## Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Git](https://git-scm.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/implicit-vibecode.git
   cd implicit-vibecode
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using Python's built-in venv
   # python -m venv .venv
   # source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   make install
   ```

## Usage

### 1. Training a Model

Train a model with default parameters:
```bash
make train
```

Customize training parameters:
```bash
python -m examples.train_and_evaluate \
  --factors 128 \
  --regularization 0.05 \
  --iterations 20 \
  --seed 42
```

### 2. Tracking Experiments

Start the MLflow UI to track experiments:
```bash
make mlflow-ui
```
Then open your browser to `http://localhost:5000`

### 3. Serving the Model

Start the FastAPI server:
```bash
make serve
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`

### 4. Making Recommendations

Get recommendations for a user:
```bash
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "k": 5}'
```

Find similar items:
```bash
curl -X 'POST' \
  'http://localhost:8000/similar' \
  -H 'Content-Type: application/json' \
  -d '{"item_id": 1, "k": 5}'
```

## Project Structure

```
implicit-vibecode/
├── examples/               # Example scripts
│   ├── train_and_evaluate.py  # Training and evaluation script
│   └── serve_model.py         # Model serving script
├── src/                    # Source code
│   └── implicit_vibecode/  # Main package
│       ├── data/           # Data loading and preprocessing
│       ├── models/         # Model implementations
│       ├── training/       # Training pipelines
│       ├── evaluation/     # Evaluation metrics
│       ├── api/            # FastAPI application
│       └── utils/          # Helper functions
├── tests/                  # Unit tests
├── requirements/           # Dependency files
├── mlruns/                 # MLflow experiment tracking
├── Makefile                # Common tasks
└── pyproject.toml          # Project configuration
```

## Development

### Running Tests
```bash
make test
```

### Code Formatting and Linting
```bash
make format  # Auto-format code
make lint    # Check code style and type hints
```

### Clean Up
```bash
make clean  # Remove cache files and build artifacts
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Implicit](https://github.com/benfred/implicit) - Fast Python Collaborative Filtering
- [MLflow](https://mlflow.org/) - Open Source Platform for the Machine Learning Lifecycle
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for APIs
- [MovieLens](https://grouplens.org/datasets/movielens/) - Sample dataset
