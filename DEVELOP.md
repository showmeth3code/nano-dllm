# Nano-VLLM Developer Guide

This document provides an overview of the nano-vllm project architecture, component descriptions, and a catalog of utility scripts for testing and debugging.

## Project Overview

Nano-VLLM is a lightweight PyTorch-based implementation of the vLLM inference engine, designed for running Large Language Models efficiently. It focuses on being clear, educational, and compatible with Hugging Face models, particularly the Qwen3 architecture.

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  ┌─────────────┐                                                   │
│  │             │                                                   │
│  │   Client    │                                                   │
│  │             │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                          │
│         │ Prompts & Sampling Parameters                            │
│         ▼                                                          │
│  ┌──────────────┐     ┌───────────────┐     ┌───────────────────┐  │
│  │              │     │               │     │                   │  │
│  │  LLM Engine  │────▶│  Scheduler    │────▶│  Block Manager    │  │
│  │              │     │               │     │                   │  │
│  └──────┬───────┘     └───────────────┘     └───────────────────┘  │
│         │                                                          │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐                                                  │
│  │              │                                                  │
│  │ Model Runner │                                                  │
│  │              │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                          │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐     ┌───────────────┐      ┌──────────────────┐  │
│  │              │     │               │      │                  │  │
│  │ Qwen3 Model  │────▶│  Layers       │─────▶│  Sampling        │  │
│  │              │     │  - Attention  │      │  - Temperature   │  │
│  └──────────────┘     │  - Linear     │      │  - Repetition    │  │
│                       │  - Rotary     │      │    Penalty       │  │
│                       └───────────────┘      └──────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Engine Layer

1. **LLMEngine** (`nanovllm/engine/llm_engine.py`)
   - Main entry point for client code
   - Manages the generation process, tokenization, and configuration
   - Coordinates other components (scheduler, model runner)

2. **Scheduler** (`nanovllm/engine/scheduler.py`)
   - Controls sequence scheduling and batching logic
   - Determines which sequences to process in each step 

3. **ModelRunner** (`nanovllm/engine/model_runner.py`)
   - Handles model execution, device management, and weight loading
   - Implements inference logic for prefill and decode phases
   - Manages KV caching and repetition penalty

4. **Sequence** (`nanovllm/engine/sequence.py`)
   - Represents a generation sequence with associated state
   - Tracks tokens, positions, and completion status

5. **BlockManager** (`nanovllm/engine/block_manager.py`)
   - Manages memory blocks for generation
   - Handles continuous batching strategy

### Model Layer

1. **Qwen3ForCausalLM** (`nanovllm/models/qwen3.py`)
   - Implementation of Qwen3 model architecture
   - Includes embedding layer, decoder layers, and output projection

2. **Layers** (`nanovllm/layers/`)
   - `attention.py`: Implements multi-head attention with grouped-query attention support
   - `linear.py`: Provides tensor-parallel linear layers
   - `rotary_embedding.py`: Implements rotary positional embeddings (RoPE)
   - `activation.py`: Activation functions like SiLU
   - `layernorm.py`: RMSNorm implementation
   - `embed_head.py`: Embedding and output layers

### Inference Layer

1. **Sampler** (`nanovllm/layers/sampler.py`)
   - Handles token sampling with temperature
   - Implements various decoding strategies
   - Includes repetition penalty

2. **Temperature** (`nanovllm/layers/temperature.py`)
   - Manages temperature scheduling during generation

3. **Config** (`nanovllm/config.py`)
   - Configuration settings for the model and engine
   - Loads and adapts Hugging Face configurations

4. **SamplingParams** (`nanovllm/sampling_params.py`)
   - Defines parameters for token sampling
   - Controls temperature, top-p, top-k, etc.

## Test and Debug Scripts Catalog

### Main Examples

1. **example.py**
   - Primary example script demonstrating model usage
   - Runs simple generation with configurable prompts
   - Tests on both CPU and MPS devices

### Model Debug Scripts

2. **debug_attention.py**
   - Tests attention module in isolation 
   - Validates key/query/value projections and attention computation

3. **debug_attention_mask.py**
   - Validates causal masking in attention
   - Tests attention with different sequence lengths

4. **debug_attention_comparison.py**
   - Compares attention output with Hugging Face reference

5. **debug_attention_detailed.py**
   - Detailed diagnostics of attention computation
   - Visualizes attention patterns

6. **debug_causal_mask.py**
   - Tests causal masking specifically

7. **debug_rotary.py**
   - Tests rotary position embeddings
   - Compares against Hugging Face RoPE implementation

8. **debug_forward_pass.py**
   - Step-by-step validation of model's forward pass
   - Compares activations at each layer

9. **debug_model_weights.py**
   - Verifies weight loading and tensor shapes
   - Compares statistics with Hugging Face weights

10. **debug_tokenizer.py**
    - Tests tokenizer functionality and integration
    - Validates encoding/decoding of various inputs

11. **debug_model_consistency.py**
    - Tests model output consistency across multiple runs
    - Validates deterministic behavior

12. **debug_vocab_size.py**
    - Tests vocabulary size handling
    - Checks embedding dimension compatibility

13. **debug_heads.py**
    - Tests individual attention heads behavior
    - Validates head dimension calculations

### Integration Test Scripts

14. **test_nano_vllm.py**
    - Main test suite for model validation
    - Comprehensive comparison with Hugging Face models
    - Tests weight loading, activation values, and text generation

15. **test_engine_debug.py**
    - Directly tests the LLMEngine component
    - Provides detailed logging of engine behavior

16. **test_hf_direct.py**
    - Tests Hugging Face model directly for comparison
    - Generates token-by-token output for reference

17. **test_hf_manual.py**
    - Manual token-by-token generation with Hugging Face
    - Used as a reference for debugging

18. **test_just_model.py**
    - Tests only the model component in isolation
    - Validates forward pass without engine

19. **test_kv_cache.py**
    - Tests key-value cache implementation
    - Validates cache behavior during generation

20. **test_model_outputs.py**
    - Tests model output structure and format
    - Compares logits and hidden states

21. **test_cache_allocation.py**
    - Tests memory allocation for KV cache
    - Validates efficient memory usage

22. **test_causal_mask_fix.py**
    - Tests fixes for causal masking issues
    - Validates correct token visibility

23. **test_fixed_model.py**
    - Tests model after specific fixes
    - Validates regression tests

24. **test_generation.py** / **test_generation_after_fix.py**
    - Tests generation capabilities
    - Compares output before and after fixes

25. **test_official_hf.py**
    - Tests against official Hugging Face API
    - Validates compatibility with upstream

### Utility Scripts

26. **generate_text.py**
    - Simplified text generation script
    - Useful for quick testing

27. **download_hf_model.py**
    - Downloads model weights from Hugging Face Hub
    - Handles caching and versioning

28. **compare_model_outputs.py**
    - Compares outputs between nano-vllm and Hugging Face
    - Provides detailed metrics on differences

29. **compare_outputs.py**
    - Simple utility to compare text outputs
    - Useful for regression testing

30. **bench.py**
    - Benchmarks performance metrics
    - Tests throughput and latency

31. **inspect_output_structure.py**
    - Inspects output tensor structures
    - Helps debug shape mismatches

## Common Development Workflows

### Adding a New Model Architecture

1. Create a new model file in `nanovllm/models/`
2. Implement the model class following the pattern in `qwen3.py`
3. Add appropriate weight loading logic in `utils/loader.py`
4. Create test scripts to validate the implementation

### Debugging Model Output Issues

1. Use `test_model_outputs.py` to validate general output structure
2. Compare with Hugging Face using `test_hf_direct.py`
3. Inspect activations with appropriate debug scripts
4. Test generation with `test_engine_debug.py`

### Improving Sampling & Generation

1. Modify `sampler.py` and/or `temperature.py`
2. Test changes with `test_generation.py`
3. Compare quality using `compare_outputs.py`
4. Benchmark performance impact with `bench.py`

## Development Tips

1. **Device Management**: Be careful with tensor device placement, especially when moving between CPU and MPS/CUDA.

2. **Shape Debugging**: Many issues arise from tensor shape mismatches. Use `print` statements or debugging scripts to validate shapes.

3. **Weight Validation**: Always verify loaded weights match Hugging Face reference using comparison scripts.

4. **Repetition Issues**: When facing repetitive output, check sampling logic and implement/tune repetition penalties.

5. **Performance Bottlenecks**: Use profiling to identify slowdowns. Common issues include unnecessary tensor copies and device transfers.
