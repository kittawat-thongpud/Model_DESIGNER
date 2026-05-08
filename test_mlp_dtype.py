#!/usr/bin/env python3
"""Test MLP dtype handling in decoder forward."""

import sys
sys.path.insert(0, '/home/rase/kittawat_ws/Model_DESIGNER')

# Mock CUDA availability for testing without GPU
import torch
from torch.nn import Module

# Test MLP class behavior
try:
    from ultralytics.nn.modules.transformer import MLP
    print("✅ MLP imported from ultralytics")
    
    # Create MLP instance
    mlp = MLP(4, 8, 4, num_layers=2)  # 4->8->4
    print(f"   MLP type: {type(mlp)}")
    print(f"   Has weight attr: {hasattr(mlp, 'weight')}")
    
    # Try to access subscriptable
    try:
        first_layer = mlp[0]
        print(f"   ✅ MLP is subscriptable: {type(first_layer)}")
        print(f"   First layer has weight: {hasattr(first_layer, 'weight')}")
    except TypeError as e:
        print(f"   ❌ MLP is NOT subscriptable: {e}")
        
    # Test forward with different dtypes
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(2, 4, dtype=dtype)
        try:
            out = mlp(x)
            print(f"   ✅ Forward with {dtype}: output shape {out.shape}")
        except Exception as e:
            print(f"   ❌ Forward with {dtype} failed: {e}")
            
except ImportError as e:
    print(f"❌ Failed to import: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Test complete")
