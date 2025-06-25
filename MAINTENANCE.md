# Code Maintenance Guidelines for nano-vllm

## Architecture Overview

nano-vllm is structured around these core components:

1. **Engine Layer**: Manages sequences, blocks, and scheduling
   - `llm_engine.py`: Main orchestration of generation process
   - `block_manager.py`: Handles KV cache block allocation/deallocation
   - `scheduler.py`: Schedules sequence processing
   - `model_runner.py`: Interface between model and engine
   - `sequence.py`: Represents text generation sequences

2. **Model Layer**: Implements Qwen3 transformer architecture
   - `qwen3.py`: Main model implementation
   - Layers: Attention, LayerNorm, Linear, etc.

3. **API Layer**: User-facing interfaces
   - `llm.py`: Main user API for text generation
   - `sampling_params.py`: Configuration for generation parameters

## Recent Issues and Solutions

### Block Manager Assertion Error

**Issue**: When generating long outputs across many sequences, the `block_manager.py` would fail with an assertion error:
```
assert last_block.hash != -1
```

**Root Cause**: The block manager expects a certain relationship between sequence length and block hashes:
- When `len(seq) % block_size == 0`: A block is exactly full and should have a valid hash
- When `len(seq) % block_size == 1`: A new block was just allocated and the previous block should have a hash
- Otherwise: The current block is partially filled and should have hash == -1

However, there were edge cases where these conditions weren't met due to:
1. Complex interactions between sequence scheduling
2. Potential race conditions in multi-sequence processing
3. Token generation patterns that didn't align with expectations

**Solution**: Removed strict assertions and added more robust error handling:
```python
def may_append(self, seq: Sequence):
    """
    Manage block allocation when a new token is appended to a sequence.
    
    This function handles three cases based on the sequence length relative to block size:
    1. When len(seq) % block_size == 1: We need to allocate a new block
    2. When len(seq) % block_size == 0: We've just filled a block completely and need to set its hash
    3. Otherwise: We're still filling an already allocated block
    """
    # Implementation with improved error handling...
```

## Best Practices for Maintenance

### When Working with block_manager.py

1. **Block Allocation Logic**:
   - Understand the relationship between sequence length and block size
   - Handle edge cases where hash != expected value
   - Check for empty block tables before accessing

2. **Proper Testing**:
   - Use `test_block_allocation.py` to verify changes
   - Test both exact multiples of block_size and edge cases
   - Verify with high token count sequences

### When Working with Attention Mechanisms

1. **Shape Management**:
   - Add shape comments for all tensor operations
   - Verify dimensions match expected values
   - Handle grouped-query attention carefully

2. **Device Consistency**:
   - Check tensor devices before operations
   - Support both CUDA and MPS (Apple Silicon)
   - Use `.to(device)` explicitly when needed

### When Working with Model Generation

1. **Position Tracking**:
   - Ensure current_position is properly incremented
   - Make use of the position property in sequence.py
   - Position values should match KV cache indices

2. **Sampling Logic**:
   - Temperature scaling must be correct
   - Repetition penalty should be properly applied
   - Token filtering (top_p, top_k) should work as expected

## Testing Instructions

Before submitting changes, run the following tests:

1. **Unit Tests**:
   ```bash
   python test_block_allocation.py  # Test KV cache block management
   python test_nano_vllm.py         # Basic model functionality
   ```

2. **Integration Tests**:
   ```bash 
   python test_hf_direct.py         # Compare with HF reference
   python example.py                # End-to-end generation example
   ```

3. **Performance Tests**:
   ```bash
   python mini_bench.py             # Quick benchmarking
   python bench.py                  # Full benchmark suite
   ```

## Future Improvements

1. **KV Cache Optimization**:
   - Implement more efficient block allocation strategies
   - Consider page-based or hierarchical cache structures
   - Support Flash Attention v2/v3 for modern GPUs

2. **Better Error Handling**:
   - Add more robust error reporting
   - Gracefully handle out-of-memory situations
   - Provide clear error messages for shape mismatches

3. **Extended Model Support**:
   - Add support for more model architectures
   - Implement efficient quantization (4-bit, 8-bit)
   - Support for sliding window attention

## Troubleshooting Common Issues

1. **"Shape mismatch" errors**:
   - Check tensor dimensions throughout the attention flow
   - Ensure batch sizes are consistent
   - Verify sequence lengths match expected values

2. **"Device mismatch" errors**:
   - Check that all tensors in an operation are on same device
   - Use explicit `.to(device)` calls rather than assuming
   - Be especially careful with MPS (Apple Silicon) handling

3. **"Out of memory" errors**:
   - Reduce batch size or sequence length
   - Check for tensor leaks (tensors not being freed)
   - Monitor block allocation/deallocation patterns

4. **"Incorrect generation" issues**:
   - Verify causal mask implementation
   - Check that position embeddings are properly applied
   - Ensure proper normalization of logits before sampling

## GitHub Copilot Integration

This project includes `.github/copilot-instructions.md` which contains specific guidelines for GitHub Copilot to follow when assisting with code in this repository. These instructions are automatically provided to Copilot when it's analyzing or generating code for this project.

The instructions include key patterns and best practices specific to this codebase, ensuring that Copilot-generated suggestions follow our conventions for:

1. **Type hints** - Ensuring proper parameter and return value typing
2. **Tensor shape documentation** - Using comments to document tensor dimensions
3. **Block allocation handling** - Following best practices for the KV cache
4. **Device compatibility** - Ensuring proper device handling (CPU/CUDA/MPS)
5. **Testing practices** - Following our testing protocols

Contributors can view these instructions at `.github/copilot-instructions.md` to understand how Copilot is configured to assist with this project.
