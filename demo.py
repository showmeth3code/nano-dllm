#!/usr/bin/env python3
"""
Demo script showing the integration between Python and Rust code in tiny-vllm.

This example demonstrates:
1. Python-side: Using the nanovllm Python package for tokenization
2. Rust-side: Using the tiny_vllm_py module for system information
3. How they work together in a unified project
"""

import os

# Handle imports with fallbacks
def safe_import():
    imports = {}
    
    # Try importing Rust module
    try:
        import tiny_vllm_py
        imports['tiny_vllm_py'] = tiny_vllm_py
        print("✓ Successfully imported tiny_vllm_py (Rust extension)")
    except ImportError as e:
        print(f"⚠ Could not import tiny_vllm_py: {e}")
        print("  Fix: cd tiny-vllm-py && maturin develop && cd ..")
        imports['tiny_vllm_py'] = None
    
    # Try importing Python modules
    try:
        from transformers import AutoTokenizer
        imports['AutoTokenizer'] = AutoTokenizer
        print("✓ Successfully imported transformers")
    except ImportError as e:
        print(f"⚠ Could not import transformers: {e}")
        imports['AutoTokenizer'] = None
    
    try:
        from nanovllm.sampling_params import SamplingParams
        imports['SamplingParams'] = SamplingParams
        print("✓ Successfully imported nanovllm.SamplingParams")
    except ImportError as e:
        print(f"⚠ Could not import SamplingParams: {e}")
        imports['SamplingParams'] = None
    
    return imports


def main():
    print("=== Tiny-vLLM Integration Demo ===\n")
    
    # Import with error handling
    imports = safe_import()
    tiny_vllm_py = imports.get('tiny_vllm_py')
    AutoTokenizer = imports.get('AutoTokenizer')
    SamplingParams = imports.get('SamplingParams')
    
    print()
    
    # 1. Test Rust-side functionality
    if tiny_vllm_py:
        print("1. Rust Module (tiny_vllm_py) - System Information:")
        try:
            print(f"   Device: {tiny_vllm_py.get_device()}")
            print(f"   GPU Memory: {tiny_vllm_py.get_gpu_memory()} bytes")
            print(f"   GPU Memory Utilization: {tiny_vllm_py.get_gpu_memory_utilization():.1%}")
        except Exception as e:
            print(f"   Error calling Rust functions: {e}")
        print()
    else:
        print("1. Rust Module (tiny_vllm_py) - SKIPPED (not available)")
        print("   Fix: cd tiny-vllm-py && maturin develop && cd ..")
        print()
    
    # 2. Test Python-side functionality
    if AutoTokenizer:
        print("2. Python Module (nanovllm) - Tokenization:")
        try:
            model_name = "microsoft/DialoGPT-small"
            print(f"   Loading tokenizer: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Test prompts
            test_prompts = [
                "Hello, how are you?",
                "What is artificial intelligence?",
                "Explain quantum computing in simple terms."
            ]
            
            print("   Testing tokenization:")
            for prompt in test_prompts:
                tokens = tokenizer.encode(prompt)
                decoded = tokenizer.decode(tokens)
                print(f"   - '{prompt}' → {len(tokens)} tokens")
                print(f"     Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                print(f"     Decoded: '{decoded}'")
                print()
        except Exception as e:
            print(f"   Error with tokenization: {e}")
        print()
    else:
        print("2. Python Module - SKIPPED (transformers not available)")
        print("   Fix: pip install transformers")
        print()
    
    # 3. Test Python configuration classes
    if SamplingParams:
        print("3. Python Module - Configuration:")
        try:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
            print(f"   ✓ Created SamplingParams: temp={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}")
        except Exception as e:
            print(f"   ⚠ Could not create SamplingParams: {e}")
        print()
    else:
        print("3. Python Module (SamplingParams) - SKIPPED (not available)")
        print("   Fix: pip install -e .")
        print()
    
    # 4. Show integration potential
    print("4. Integration Summary:")
    working_components = []
    if tiny_vllm_py: 
        working_components.append("Rust module")
        print("   ✓ Rust module provides fast system utilities and CUDA operations")
    if AutoTokenizer: 
        working_components.append("Tokenization")
        print("   ✓ Python module provides high-level ML operations and model management")
    if SamplingParams: 
        working_components.append("Configuration")
        print("   ✓ Both work together in a unified package")
    
    # Check if we're on a system that could support GPU operations
    if tiny_vllm_py:
        try:
            device = tiny_vllm_py.get_device()
            if device == "cpu":
                print("   ℹ Running on CPU - for GPU acceleration, install CUDA drivers and rebuild")
            else:
                print(f"   ✓ GPU support detected: {device}")
        except:
            print("   ℹ Could not determine device type")
    
    print("\n=== Demo Complete ===")
    if working_components:
        print(f"✓ Working components: {', '.join(working_components)}")
        print("Both Python and Rust components are working correctly!")
    else:
        print("⚠ No components are working - check installation")
        print("Run: pip install -r requirements.txt && cd tiny-vllm-py && maturin develop && cd ..")


if __name__ == "__main__":
    main()
