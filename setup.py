"""Setup script for the implicit-vibecode package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="implicit-vibecode",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A recommender system using implicit feedback with ALS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/implicit-vibecode",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "implicit>=0.7.0",
        "mlflow>=2.7.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pydantic>=1.10.0",
        "category-encoders>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            "sphinx>=6.1.3",
            "sphinx-rtd-theme>=1.2.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
