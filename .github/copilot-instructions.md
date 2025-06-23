We are working with a PyTorch-based LLM inference engine (nano-vllm) focused on efficient text generation for Qwen3 models with optimized attention and caching mechanisms.

Always include proper type hints for function parameters and return values in Python code to maintain clarity and compatibility.

Use tensor shape comments for all tensor operations in PyTorch. Include comments like `# [batch_size, seq_len, hidden_dim]` before operating on tensors.

When dealing with block allocation in KV cache, be aware of the relationship between sequence length and block_size. Avoid strict assertions and handle edge cases gracefully.

For debugging, prefer adding explicit validation steps and informative error messages rather than using assertions that might fail silently.

Always check tensor devices before operations to ensure compatibility between CPU, CUDA, and MPS (Apple Silicon).

When implementing grouped-query attention, verify shapes after reshaping operations and ensure proper handling of causal masks.

Use `test_block_allocation.py` to verify KV cache block manager logic and `test_nano_vllm.py` for model integration testing.

Our benchmarking follows a specific pattern: start with small batches in mini_bench.py before scaling to full tests in bench.py.

For sampling logic implementation, ensure temperature scaling is correct and repetition penalty is properly applied across batch dimensions.

Document all complex algorithms and non-obvious optimizations, especially in the block manager and attention implementation.

When tracking sequence positions, ensure the position property in sequence.py is correctly used and position values match KV cache indices.
