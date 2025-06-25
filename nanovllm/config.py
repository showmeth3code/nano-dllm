from transformers import AutoConfig, AutoTokenizer

class Config:
    """
    Configuration for the nano-vllm models.
    
    This class handles the configuration for the models, including loading
    the HuggingFace config and setting up additional parameters.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Load HuggingFace config with trust_remote_code=True for Qwen models
        self.hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Set default values
        self.kvcache_block_size = 16
        self.max_num_seqs = 256
        self.enforce_eager = False
        self.tensor_parallel_size = 1
        self.max_num_batched_tokens = 4096  # Default value, increased from 1024
        self.num_kvcache_blocks = 4096  # Default value, increased from 1024
        
        # Get the vocab size from the model config
        self.vocab_size = getattr(self.hf_config, "vocab_size", None)
        
        # Load tokenizer to get EOS token ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.eos = self.tokenizer.eos_token_id
        
        # Print model info for debugging
        self._print_model_info()
        
    def _print_model_info(self):
        """Print important model configuration information for debugging."""
        print("=== MODEL CONFIGURATION ===")
        print(f"Model path: {self.model_path}")
        print(f"Model type: {self.hf_config.model_type}")
        print(f"Vocab size: {self.hf_config.vocab_size}")
        print(f"Hidden size: {self.hf_config.hidden_size}")
        print(f"Num attention heads: {self.hf_config.num_attention_heads}")
        
        # Print num_key_value_heads if available
        num_kv_heads = getattr(self.hf_config, "num_key_value_heads", self.hf_config.num_attention_heads)
        print(f"Num key-value heads: {num_kv_heads}")
        
        # Print tokenizer info
        print(f"Tokenizer class: {self.tokenizer.__class__.__name__}")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"EOS token ID: {self.eos}")
        print("===============================")

    @property
    def vocab_size(self):
        return self.hf_config.vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self.hf_config.vocab_size = value
