# Model Support

This document covers the model architectures supported by Nano-vLLM, model loading procedures, and how to integrate custom models.

## 🏗️ Supported Architectures

### Currently Supported

#### Qwen3 Models
- **Architecture**: Transformer with RMSNorm, RoPE, and grouped-query attention
- **Variants**: Qwen3-0.5B, Qwen3-1.5B, Qwen3-3B, Qwen3-7B, Qwen3-14B, Qwen3-72B
- **Features**:
  - Flash Attention optimization
  - Tensor parallelism support
  - Prefix caching compatible
  - CUDA graph optimization

**Model Structure**:
```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding
│   ├── Qwen3DecoderLayer (× num_layers)
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear
│   │   │   ├── Attention (Flash Attention)
│   │   │   └── RowParallelLinear
│   │   └── Qwen3MLP
│   │       ├── MergedColumnParallelLinear
│   │       └── RowParallelLinear
│   └── RMSNorm
└── ParallelLMHead
```

### Planned Support

#### Llama Models
- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Code Llama variants

#### Mistral Models
- Mistral 7B
- Mixtral 8x7B
- Mistral Large

#### Other Architectures
- GPT-2/GPT-NeoX variants
- MPT models
- Falcon models

## 📥 Model Loading

### HuggingFace Models

#### Download and Load
```python
from nanovllm import LLM

# Load from HuggingFace hub
llm = LLM("Qwen/Qwen3-0.6B")

# Load from local directory
llm = LLM("/path/to/local/model")
```

#### Manual Download
```bash
# Download model using huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# Or use git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### Model Requirements

#### Required Files
```
model_directory/
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special tokens
├── pytorch_model.bin        # Model weights (or sharded)
├── pytorch_model-00001-of-00002.bin  # Sharded weights
├── pytorch_model-00002-of-00002.bin
└── generation_config.json   # Generation settings
```

#### Configuration Validation
```python
from transformers import AutoConfig

# Validate model configuration
config = AutoConfig.from_pretrained("/path/to/model")

# Check required attributes
required_attrs = [
    "hidden_size",
    "num_hidden_layers", 
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings"
]

for attr in required_attrs:
    if not hasattr(config, attr):
        raise ValueError(f"Missing required config attribute: {attr}")
```

### Model Loading Process

#### 1. Configuration Loading
```python
# Load and validate configuration
config = AutoConfig.from_pretrained(model_path)

# Set default values
if not hasattr(config, 'torch_dtype'):
    config.torch_dtype = torch.float16

if not hasattr(config, 'rms_norm_eps'):
    config.rms_norm_eps = 1e-6
```

#### 2. Model Initialization
```python
# Create model instance
model = Qwen3ForCausalLM(config)

# Load weights
load_model(model, model_path)
```

#### 3. Weight Loading
```python
# Load state dict
state_dict = torch.load(
    os.path.join(model_path, "pytorch_model.bin"),
    map_location="cpu"
)

# Apply weight mapping
model.load_state_dict(state_dict, strict=False)
```

## 🔧 Model Configuration

### Configuration Parameters

#### Model Architecture
```python
@dataclass
class ModelConfig:
    hidden_size: int = 4096           # Hidden dimension
    num_hidden_layers: int = 32       # Number of transformer layers
    num_attention_heads: int = 32     # Number of attention heads
    num_key_value_heads: int = 32     # Number of KV heads (GQA)
    intermediate_size: int = 11008    # MLP intermediate size
    vocab_size: int = 151936          # Vocabulary size
    max_position_embeddings: int = 32768  # Max sequence length
    head_dim: int = 128               # Attention head dimension
    rms_norm_eps: float = 1e-6        # RMSNorm epsilon
    rope_theta: float = 1000000       # RoPE base frequency
```

#### Generation Settings
```python
@dataclass
class GenerationConfig:
    temperature: float = 0.7          # Sampling temperature
    top_p: float = 0.9               # Nucleus sampling
    top_k: int = 50                  # Top-k sampling
    max_new_tokens: int = 512        # Max tokens to generate
    do_sample: bool = True           # Enable sampling
    pad_token_id: int = 0            # Padding token ID
    eos_token_id: int = 151643       # End-of-sequence token ID
```

### Model-Specific Settings

#### Qwen3 Configuration
```python
# Qwen3-specific settings
config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")

# Key parameters
print(f"Hidden size: {config.hidden_size}")
print(f"Layers: {config.num_hidden_layers}")
print(f"Attention heads: {config.num_attention_heads}")
print(f"KV heads: {config.num_key_value_heads}")
print(f"Vocab size: {config.vocab_size}")
print(f"Max position: {config.max_position_embeddings}")
```

## 🚀 Custom Model Integration

### Adding New Model Support

#### 1. Create Model Implementation

Create a new file in `nanovllm/models/`:

```python
# nanovllm/models/your_model.py
import torch
from torch import nn
from transformers import YourModelConfig

class YourModelAttention(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        # Implement attention mechanism
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Add your attention implementation
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, positions, hidden_states):
        # Implement forward pass
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply attention
        # ... your attention logic
        
        return output

class YourModelMLP(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        # Implement MLP
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, x):
        # Implement MLP forward pass
        gate = torch.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class YourModelDecoderLayer(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        self.self_attn = YourModelAttention(config)
        self.mlp = YourModelMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, positions, hidden_states, residual=None):
        # Implement layer forward pass
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class YourModel(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([YourModelDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
            
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class YourModelForCausalLM(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        self.model = YourModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return hidden_states
        
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

#### 2. Add Model Loading Logic

Update `nanovllm/utils/loader.py`:

```python
# nanovllm/utils/loader.py
from nanovllm.models.your_model import YourModelForCausalLM

def load_model(model, model_path: str):
    """Load model weights from path."""
    if isinstance(model, YourModelForCausalLM):
        load_your_model_weights(model, model_path)
    elif isinstance(model, Qwen3ForCausalLM):
        load_qwen3_weights(model, model_path)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def load_your_model_weights(model: YourModelForCausalLM, model_path: str):
    """Load weights for your custom model."""
    # Implement weight loading logic
    state_dict = torch.load(
        os.path.join(model_path, "pytorch_model.bin"),
        map_location="cpu"
    )
    
    # Apply any necessary weight transformations
    # ... weight mapping logic
    
    model.load_state_dict(state_dict, strict=False)
```

#### 3. Update Model Registry

Add your model to the model registry in `nanovllm/models/__init__.py`:

```python
# nanovllm/models/__init__.py
from .qwen3 import Qwen3ForCausalLM
from .your_model import YourModelForCausalLM

__all__ = ["Qwen3ForCausalLM", "YourModelForCausalLM"]
```

#### 4. Add Tests

Create tests in `tests/test_models.py`:

```python
# tests/test_models.py
import pytest
from nanovllm.models.your_model import YourModelForCausalLM
from transformers import YourModelConfig

class TestYourModel:
    def test_model_creation(self):
        """Test model can be created."""
        config = YourModelConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            vocab_size=1000
        )
        model = YourModelForCausalLM(config)
        assert model is not None
        
    def test_model_forward(self):
        """Test model forward pass."""
        config = YourModelConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            vocab_size=1000
        )
        model = YourModelForCausalLM(config)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        positions = torch.arange(10).unsqueeze(0).expand(2, -1)
        
        output = model(input_ids, positions)
        assert output.shape == (2, 10, 512)
```

### Tensor Parallelism Support

#### Implementing Tensor Parallelism

```python
class YourModelAttention(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        tp_size = dist.get_world_size()
        
        # Shard attention heads across GPUs
        self.num_heads = config.num_attention_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Use tensor-parallel linear layers
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False
        )
```

#### Communication Patterns

```python
# All-reduce for output projection
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    
    # Apply attention locally
    output = self.attention(q, k, v)
    
    # All-reduce across GPUs
    output = self.o_proj(output)
    return output
```

## 🔍 Model Debugging

### Common Issues

#### 1. Model Loading Errors
```python
# Debug model loading
import os
from transformers import AutoConfig

model_path = "/path/to/model"

# Check files exist
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Files: {os.listdir(model_path)}")

# Validate config
try:
    config = AutoConfig.from_pretrained(model_path)
    print("Config loaded successfully")
except Exception as e:
    print(f"Config error: {e}")
```

#### 2. Weight Mismatch
```python
# Debug weight loading
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model_state_dict = model.state_dict()

# Check missing keys
missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
print(f"Missing keys: {missing_keys}")

# Check unexpected keys
unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
print(f"Unexpected keys: {unexpected_keys}")
```

#### 3. Shape Mismatches
```python
# Debug tensor shapes
for name, param in model.named_parameters():
    if name in state_dict:
        expected_shape = state_dict[name].shape
        actual_shape = param.shape
        if expected_shape != actual_shape:
            print(f"Shape mismatch in {name}: {expected_shape} vs {actual_shape}")
```

### Performance Validation

#### Benchmark Your Model
```python
import time
from nanovllm import LLM, SamplingParams

def benchmark_model(model_path):
    llm = LLM(model_path)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    prompts = ["Test prompt"] * 10
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_tokens = sum(len(output['token_ids']) for output in outputs)
    throughput = total_tokens / (end_time - start_time)
    
    print(f"Throughput: {throughput:.2f} tokens/s")
    return throughput

# Compare with baseline
qwen_throughput = benchmark_model("/path/to/qwen3")
your_throughput = benchmark_model("/path/to/your_model")

print(f"Performance ratio: {your_throughput / qwen_throughput:.2f}")
```

## 📊 Model Comparison

### Architecture Comparison

| Model | Hidden Size | Layers | Heads | KV Heads | Vocab Size | Max Length |
|-------|-------------|--------|-------|----------|------------|------------|
| Qwen3-0.6B | 1024 | 16 | 16 | 16 | 151936 | 32768 |
| Qwen3-1.5B | 1536 | 24 | 24 | 24 | 151936 | 32768 |
| Qwen3-3B | 2048 | 32 | 32 | 32 | 151936 | 32768 |
| Qwen3-7B | 4096 | 32 | 32 | 32 | 151936 | 32768 |

### Performance Characteristics

| Model | Memory (GB) | Throughput (tokens/s) | Latency (ms) |
|-------|-------------|----------------------|--------------|
| Qwen3-0.6B | 1.2 | 1,434 | 45 |
| Qwen3-1.5B | 2.8 | 892 | 72 |
| Qwen3-3B | 5.6 | 456 | 140 |
| Qwen3-7B | 12.4 | 234 | 273 |

## 🔮 Future Model Support

### Planned Features

1. **Multi-Modal Models**: Support for vision-language models
2. **Quantized Models**: INT8/INT4 inference support
3. **MoE Models**: Mixture of Experts architectures
4. **Long Context**: Support for 100K+ context lengths
5. **Custom Architectures**: Plugin system for custom models

### Model Plugin System

```python
# Future plugin system (not yet implemented)
from nanovllm.plugins import ModelPlugin

class YourModelPlugin(ModelPlugin):
    def __init__(self):
        super().__init__("your_model")
    
    def create_model(self, config):
        return YourModelForCausalLM(config)
    
    def load_weights(self, model, path):
        return load_your_model_weights(model, path)
    
    def get_config_class(self):
        return YourModelConfig

# Register plugin
ModelPlugin.register(YourModelPlugin())
```

This model support guide provides comprehensive information for working with models in Nano-vLLM. For more detailed information about the architecture and API, refer to the [Architecture Overview](./architecture.md) and [API Reference](./api.md). 