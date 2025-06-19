import tiny_vllm_py

# Expose constants from the Rust implementation.
DEFAULT_MAX_NUM_BATCHED_TOKENS = tiny_vllm_py.default_max_num_batched_tokens()
DEFAULT_MAX_NUM_SEQS = tiny_vllm_py.default_max_num_seqs()
DEFAULT_MAX_MODEL_LEN = tiny_vllm_py.default_max_model_len()
DEFAULT_GPU_MEMORY_UTILIZATION = tiny_vllm_py.default_gpu_memory_utilization()
DEFAULT_TENSOR_PARALLEL_SIZE = tiny_vllm_py.default_tensor_parallel_size()
DEFAULT_ENFORCE_EAGER = tiny_vllm_py.default_enforce_eager()
DEFAULT_KVCACHE_BLOCK_SIZE = tiny_vllm_py.default_kvcache_block_size()
DEFAULT_NUM_KVCACHE_BLOCKS = tiny_vllm_py.default_num_kvcache_blocks()
DEFAULT_EOS = tiny_vllm_py.default_eos()
