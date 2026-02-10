"""
Quick Test Script - Verify model configuration
"""

from model_config import MODELS, CRITERIA, get_model_names, get_criteria_weights

def test_configuration():
    print("="*70)
    print("CONFIGURATION TEST")
    print("="*70)
    
    # Test 1: Models
    print("\n✓ TEST 1: Model Configuration")
    print(f"   Total models: {len(MODELS)}")
    print(f"   Model names: {', '.join(get_model_names())}")
    
    # Test 2: Criteria
    print("\n✓ TEST 2: TOPSIS Criteria")
    total_weight = sum(get_criteria_weights())
    print(f"   Total criteria: {len(CRITERIA)}")
    print(f"   Total weight: {total_weight*100}%")
    
    if abs(total_weight - 1.0) < 0.001:
        print("   ✓ Weights sum to 100%")
    else:
        print(f"   ✗ WARNING: Weights sum to {total_weight*100}%")
    
    # Test 3: Import test
    print("\n✓ TEST 3: Package Imports")
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("   ✗ PyTorch not found")
    
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}")
    except ImportError:
        print("   ✗ Transformers not found")
    
    try:
        import sentence_transformers
        print(f"   Sentence-Transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("   ✗ Sentence-Transformers not found")
    
    try:
        import pandas as pd
        print(f"   Pandas: {pd.__version__}")
    except ImportError:
        print("   ✗ Pandas not found")
    
    try:
        import numpy as np
        print(f"   NumPy: {np.__version__}")
    except ImportError:
        print("   ✗ NumPy not found")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nReady to run: python evaluate_models.py")

if __name__ == "__main__":
    test_configuration()
