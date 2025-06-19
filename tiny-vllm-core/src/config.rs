use serde::{Deserialize, Serialize};

/// Default settings matching the Python implementation.
pub mod settings {
    /// Max tokens that can be batched together in a single forward pass.
    pub const MAX_NUM_BATCHED_TOKENS: usize = 32_768;
    /// Maximum number of sequences processed concurrently.
    pub const MAX_NUM_SEQS: usize = 512;
    /// Maximum model sequence length.
    pub const MAX_MODEL_LEN: usize = 4096;
    /// Fraction of GPU memory the engine is allowed to utilize.
    pub const GPU_MEMORY_UTILIZATION: f32 = 0.9;
    /// Tensor parallel world size.
    pub const TENSOR_PARALLEL_SIZE: usize = 1;
    /// Whether to enforce eager execution.
    pub const ENFORCE_EAGER: bool = false;
    /// KV cache block size in tokens.
    pub const KVCACHE_BLOCK_SIZE: usize = 256;
    /// Number of KV cache blocks to allocate. -1 means auto.
    pub const NUM_KVCACHE_BLOCKS: isize = -1;
    /// Default end-of-sequence token id.
    pub const EOS: i64 = -1;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VllmConfig {
    pub model: String,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub gpu_memory_utilization: f32,
    pub tensor_parallel_size: usize,
    pub enforce_eager: bool,
    pub eos: i64,
    pub kvcache_block_size: usize,
    pub num_kvcache_blocks: isize,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_num_batched_tokens: settings::MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: settings::MAX_NUM_SEQS,
            max_model_len: settings::MAX_MODEL_LEN,
            gpu_memory_utilization: settings::GPU_MEMORY_UTILIZATION,
            tensor_parallel_size: settings::TENSOR_PARALLEL_SIZE,
            enforce_eager: settings::ENFORCE_EAGER,
            eos: settings::EOS,
            kvcache_block_size: settings::KVCACHE_BLOCK_SIZE,
            num_kvcache_blocks: settings::NUM_KVCACHE_BLOCKS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_roundtrip() {
        let mut cfg = VllmConfig::default();
        cfg.model = "facebook/opt-125m".to_string();
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: VllmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, decoded);
    }
}
