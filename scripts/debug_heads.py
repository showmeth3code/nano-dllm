#!/usr/bin/env python3
from nanovllm.config import Config

# Check the relationship between heads and hidden size
config = Config(model="Qwen/Qwen2.5-0.5B")

print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_key_value_heads: {config.num_key_value_heads}")
print(f"hidden_size: {config.hidden_size}")
print(f"head_dim: {config.hidden_size // config.num_attention_heads}")
print(f"num_attention_heads * head_dim: {config.num_attention_heads * (config.hidden_size // config.num_attention_heads)}")
print(f"Should equal hidden_size: {config.num_attention_heads * (config.hidden_size // config.num_attention_heads) == config.hidden_size}")
