"""Test script to verify the installation and basic functionality."""
import sys
import logging

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import implicit
        import mlflow
        import fastapi
        import uvicorn
        import pydantic
        import category_encoders
        
        print("✅ All required packages are installed and can be imported.")
        return True
    except ImportError as e:
        print(f"❌ Error importing required package: {e}")
        return False

def test_installation():
    """Test the installation by importing the main package."""
    print("\nTesting package installation...")
    try:
        import implicit_vibecode
        from implicit_vibecode.models.als_model import ALSModel
        from implicit_vibecode.data.dataset import load_movielens_sample
        
        print("✅ Package installed successfully.")
        return True
    except Exception as e:
        print(f"❌ Error importing package: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Implicit Vibecode Installation")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    import_ok = test_imports()
    install_ok = test_installation()
    
    print("\nTest Summary:")
    print(f"- Required imports: {'✅' if import_ok else '❌'}")
    print(f"- Package installation: {'✅' if install_ok else '❌'}")
    
    if import_ok and install_ok:
        print("\n🎉 All tests passed! You're ready to use Implicit Vibecode.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
