# API Reference

This document provides comprehensive API documentation for Nano-vLLM, including all public classes, methods, and configuration options.

## 📦 Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 🔧 Core Classes

### LLM

The main class for text generation with Nano-vLLM.

```python
from nanovllm import LLM

class LLM:
    def __init__(
        self,
        model: str,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        kvcache_block_size: int = 256,
        num_kvcache_blocks: int = -1,
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | Required | Path to the HuggingFace model directory |
| `max_num_batched_tokens` | `int` | `16384` | Maximum number of tokens to process in a single batch |
| `max_num_seqs` | `int` | `512` | Maximum number of concurrent sequences |
| `max_model_len` | `int` | `4096` | Maximum sequence length (truncated to model's max position embeddings) |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory to use for KV cache (0.0-1.0) |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism (1-8) |
| `enforce_eager` | `bool` | `False` | Disable CUDA graph optimization |
| `kvcache_block_size` | `int` | `256` | Size of KV cache blocks (must be divisible by 256) |
| `num_kvcache_blocks` | `int` | `-1` | Number of KV cache blocks (auto-calculated if -1) |

#### Methods

##### `generate()`

Generate text from prompts using the specified sampling parameters.

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[dict]
```

**Parameters:**
- `prompts`: List of text prompts or token ID lists
- `sampling_params`: Sampling parameters (single or list for each prompt)
- `use_tqdm`: Show progress bar during generation

**Returns:**
- List of dictionaries with keys:
  - `text`: Generated text string
  - `token_ids`: List of generated token IDs

**Example:**
```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

prompts = ["Hello, how are you?", "What is machine learning?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output['text']}")
```

##### `add_request()`

Add a single request to the generation queue.

```python
def add_request(
    self,
    prompt: str | list[int],
    sampling_params: SamplingParams,
)
```

**Parameters:**
- `prompt`: Text prompt or token ID list
- `sampling_params`: Sampling parameters for this request

##### `step()`

Execute one generation step (for manual control).

```python
def step() -> tuple[list[tuple], int]
```

**Returns:**
- Tuple of (outputs, num_tokens)
  - `outputs`: List of (seq_id, token_ids) tuples for completed sequences
  - `num_tokens`: Number of tokens processed (positive for prefill, negative for decode)

##### `is_finished()`

Check if all requests have been completed.

```python
def is_finished() -> bool
```

**Returns:**
- `True` if all requests are finished, `False` otherwise

### SamplingParams

Configuration class for text generation parameters.

```python
from nanovllm import SamplingParams

class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `1.0` | Sampling temperature (0.0 = deterministic, higher = more random) |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter (0.0-1.0) |
| `top_k` | `int` | `-1` | Top-k sampling (use all tokens if -1) |
| `max_tokens` | `int` | `16` | Maximum number of tokens to generate |
| `ignore_eos` | `bool` | `False` | Whether to ignore end-of-sequence tokens |

#### Examples

```python
# Deterministic generation
deterministic = SamplingParams(temperature=0.0, max_tokens=50)

# Creative generation
creative = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)

# Short, focused generation
focused = SamplingParams(temperature=0.3, top_k=10, max_tokens=20)
```

## 🔧 Configuration

### Config

Internal configuration class used by the engine.

```python
from nanovllm.config import Config

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
```

## 📝 Usage Examples

### Basic Text Generation

```python
from nanovllm import LLM, SamplingParams

# Initialize model
llm = LLM("/path/to/qwen3-model")

# Configure generation
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

# Generate text
prompts = ["Explain quantum computing in simple terms."]
outputs = llm.generate(prompts, sampling_params)

print(outputs[0]["text"])
```

### Batch Processing

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")

# Multiple prompts with different parameters
prompts = [
    "Write a short story about a robot.",
    "Explain photosynthesis.",
    "What is the capital of France?"
]

sampling_params = [
    SamplingParams(temperature=0.8, max_tokens=200),  # Creative story
    SamplingParams(temperature=0.3, max_tokens=150),  # Factual explanation
    SamplingParams(temperature=0.1, max_tokens=50),   # Direct answer
]

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {output['text']}\n")
```

### Manual Control

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")

# Add requests manually
llm.add_request("Start a story:", SamplingParams(temperature=0.8, max_tokens=100))
llm.add_request("Explain AI:", SamplingParams(temperature=0.3, max_tokens=80))

# Process step by step
while not llm.is_finished():
    outputs, num_tokens = llm.step()
    
    for seq_id, token_ids in outputs:
        print(f"Completed sequence {seq_id}: {len(token_ids)} tokens")
    
    print(f"Processed {num_tokens} tokens this step")
```

### Multi-GPU Setup

```python
from nanovllm import LLM, SamplingParams

# Use 2 GPUs for tensor parallelism
llm = LLM(
    "/path/to/model",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=200)
prompts = ["Generate a long response about machine learning."]

outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### Token-Level Control

```python
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Use token IDs directly
token_ids = tokenizer.encode("The future of AI is")
llm = LLM("/path/to/model")

sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
outputs = llm.generate([token_ids], sampling_params)

# Access both text and token IDs
result = outputs[0]
print(f"Text: {result['text']}")
print(f"Token IDs: {result['token_ids']}")
```

## 🔧 Advanced Configuration

### Memory Management

```python
# Optimize for memory usage
llm = LLM(
    "/path/to/model",
    gpu_memory_utilization=0.7,  # Use less GPU memory
    max_num_batched_tokens=8192,  # Smaller batches
    max_num_seqs=256,             # Fewer concurrent sequences
    kvcache_block_size=512        # Larger blocks for efficiency
)
```

### Performance Tuning

```python
# Optimize for speed
llm = LLM(
    "/path/to/model",
    enforce_eager=False,          # Enable CUDA graphs
    gpu_memory_utilization=0.95,  # Use more GPU memory
    max_num_batched_tokens=32768, # Larger batches
    max_num_seqs=1024             # More concurrent sequences
)
```

### Debug Mode

```python
# Disable optimizations for debugging
llm = LLM(
    "/path/to/model",
    enforce_eager=True,           # Disable CUDA graphs
    tensor_parallel_size=1        # Single GPU only
)
```

## 🚨 Error Handling

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce memory usage
   llm = LLM("/path/to/model", gpu_memory_utilization=0.7)
   ```

2. **Model Not Found**
   ```python
   # Ensure model path is correct
   llm = LLM("/absolute/path/to/model")
   ```

3. **CUDA Graph Errors**
   ```python
   # Disable CUDA graphs
   llm = LLM("/path/to/model", enforce_eager=True)
   ```

### Best Practices

1. **Always check model path**: Ensure the model directory exists and contains the required files
2. **Monitor memory usage**: Start with conservative `gpu_memory_utilization` values
3. **Use appropriate batch sizes**: Balance between throughput and memory usage
4. **Handle long sequences**: Consider `max_model_len` for your use case
5. **Test with small batches first**: Validate your setup before scaling up

## 🔗 Integration Examples

### With Transformers

```python
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Initialize Nano-vLLM
llm = LLM("/path/to/model")

# Use chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate([prompt], sampling_params)

print(outputs[0]["text"])
```

### With FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from nanovllm import LLM, SamplingParams

app = FastAPI()
llm = LLM("/path/to/model")

class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 100

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    return {"generated_text": outputs[0]["text"]}
```

This API reference provides comprehensive documentation for all public interfaces in Nano-vLLM. For more advanced usage patterns and internal implementation details, refer to the [Architecture Overview](./architecture.md) and [Development Guide](./development.md). 