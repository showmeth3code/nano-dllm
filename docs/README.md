# Nano-vLLM Documentation

Welcome to the Nano-vLLM documentation! This guide provides comprehensive information about the lightweight vLLM implementation built from scratch.

## 📚 Documentation Sections

### [Architecture Overview](./architecture.md)
- High-level system architecture
- Component interactions
- Design principles and patterns

### [API Reference](./api.md)
- LLM class interface
- SamplingParams configuration
- Usage examples and patterns

### [Performance Guide](./performance.md)
- Optimization techniques
- Benchmarking results
- Performance tuning tips

### [Development Guide](./development.md)
- Setup and installation
- Contributing guidelines
- Code structure and conventions

### [Model Support](./models.md)
- Supported model architectures
- Model loading and configuration
- Custom model integration

## 🚀 Quick Start

```python
from nanovllm import LLM, SamplingParams

# Initialize the model
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)

# Configure generation parameters
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Generate text
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## 🎯 Key Features

- **Fast Inference**: Comparable speeds to vLLM with optimized CUDA operations
- **Lightweight**: Clean implementation in ~1,200 lines of Python code
- **Memory Efficient**: Advanced KV cache management with prefix caching
- **Scalable**: Multi-GPU tensor parallelism support
- **Readable**: Well-documented, modular architecture

## 📊 Performance

| Metric | Value |
|--------|-------|
| Code Size | ~1,200 lines |
| Throughput | 1,434 tokens/s (RTX 4070) |
| Memory Efficiency | Block-based KV cache |
| GPU Support | Multi-GPU tensor parallelism |

## 🤝 Contributing

See the [Development Guide](./development.md) for information on contributing to Nano-vLLM.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 