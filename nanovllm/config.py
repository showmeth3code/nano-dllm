import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    is_dllm = False
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_layout: str = "unified"  # "unified" or "distinct"
    mask_token_id: int = 151666
    diffusion_block_size: int = 32
    accept_threshold: float = 0.9
    add_new_block_threshold: float = 1.0
    complete_threshold: float = 0.95
    port: int = 2444
    

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
