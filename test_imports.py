#!/usr/bin/env python3
"""Test script to verify all components are working correctly."""

def test_rust_module():
    """Test that the Rust module can be imported and basic functions work."""
    try:
        import tiny_vllm_py
        print("✓ Successfully imported tiny_vllm_py")
        
        # Test basic functions
        device = tiny_vllm_py.get_device()
        memory = tiny_vllm_py.get_gpu_memory()
        utilization = tiny_vllm_py.get_gpu_memory_utilization()
        
        print(f"  Device: {device}")
        print(f"  GPU Memory: {memory}")
        print(f"  GPU Utilization: {utilization:.1%}")
        
        return True
    except ImportError as e:
        print(f"✗ Could not import tiny_vllm_py: {e}")
        print("  Fix: cd tiny-vllm-py && maturin develop && cd ..")
        return False
    except Exception as e:
        print(f"✗ Error testing tiny_vllm_py: {e}")
        return False

def test_python_module():
    """Test that the Python modules can be imported."""
    try:
        from nanovllm.sampling_params import SamplingParams
        from nanovllm import LLM
        print("✓ Successfully imported nanovllm components")
        
        # Test creating sampling params
        params = SamplingParams(temperature=0.8)
        print(f"  Created SamplingParams with temperature: {params.temperature}")
        
        return True
    except ImportError as e:
        print(f"✗ Could not import nanovllm: {e}")
        print("  Fix: pip install -e .")
        return False
    except Exception as e:
        print(f"✗ Error testing nanovllm: {e}")
        return False

def test_transformers():
    """Test that transformers can be imported."""
    try:
        from transformers import AutoTokenizer
        print("✓ Successfully imported transformers")
        return True
    except ImportError as e:
        print(f"✗ Could not import transformers: {e}")
        print("  Fix: pip install transformers")
        return False

if __name__ == "__main__":
    print("=== Testing Tiny-vLLM Components ===\n")
    
    rust_ok = test_rust_module()
    python_ok = test_python_module()
    transformers_ok = test_transformers()
    
    print(f"\n=== Results ===")
    print(f"Rust module: {'✓ PASS' if rust_ok else '✗ FAIL'}")
    print(f"Python module: {'✓ PASS' if python_ok else '✗ FAIL'}")
    print(f"Transformers: {'✓ PASS' if transformers_ok else '✗ FAIL'}")
    
    if rust_ok and python_ok and transformers_ok:
        print("\n🎉 All tests passed!")
        print("You can now run:")
        print("  python demo.py")
        print("  python example.py")
        exit(0)
    else:
        print("\n⚠ Some tests failed - check the fixes above")
        exit(1)
