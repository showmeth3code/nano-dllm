# Development Guide

This guide provides information for developers who want to contribute to Nano-vLLM, understand the codebase, or extend its functionality.

## 🛠️ Development Setup

### Prerequisites

- **Python**: 3.10-3.12
- **CUDA**: 11.8 or later
- **GPU**: NVIDIA GPU with compute capability 7.0+
- **PyTorch**: 2.4.0 or later

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GeeeekExplorer/nano-vllm.git
   cd nano-vllm
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Verify installation**
   ```bash
   python -c "from nanovllm import LLM, SamplingParams; print('Installation successful!')"
   ```

### Development Environment

#### Recommended Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
pip install black isort mypy pytest

# Install pre-commit hooks
pre-commit install
```

#### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 📁 Code Structure

```
nanovllm/
├── __init__.py              # Public API exports
├── config.py                # Configuration management
├── llm.py                   # Main LLM class (thin wrapper)
├── sampling_params.py       # Generation parameters
├── engine/                  # Core inference engine
│   ├── __init__.py
│   ├── llm_engine.py        # Main orchestrator
│   ├── model_runner.py      # Model execution
│   ├── scheduler.py         # Request scheduling
│   ├── sequence.py          # Sequence management
│   └── block_manager.py     # KV cache management
├── layers/                  # Neural network layers
│   ├── __init__.py
│   ├── attention.py         # Flash attention
│   ├── linear.py            # Tensor-parallel linear layers
│   ├── layernorm.py         # RMS normalization
│   ├── embed_head.py        # Embedding and output projection
│   ├── rotary_embedding.py  # RoPE positional encoding
│   ├── activation.py        # Activation functions
│   └── sampler.py           # Sampling logic
├── models/                  # Model implementations
│   ├── __init__.py
│   └── qwen3.py             # Qwen3 model
└── utils/                   # Utility functions
    ├── __init__.py
    ├── context.py           # CUDA context management
    └── loader.py            # Model loading utilities
```

## 🔧 Development Workflow

### Code Style

Nano-vLLM follows strict code style guidelines:

#### Python Style
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **PEP 8**: Style guide compliance

#### Naming Conventions
- **Classes**: PascalCase (`LLMEngine`, `ModelRunner`)
- **Functions/Methods**: snake_case (`generate`, `add_request`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_NUM_SEQS`)
- **Variables**: snake_case (`token_ids`, `sampling_params`)

#### Type Hints
All public APIs must include type hints:

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[dict[str, str | list[int]]]:
    """Generate text from prompts.
    
    Args:
        prompts: List of text prompts or token ID lists
        sampling_params: Sampling parameters
        use_tqdm: Show progress bar
        
    Returns:
        List of generation results with text and token_ids
    """
```

### Testing

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nanovllm

# Run specific test file
pytest tests/test_llm.py

# Run with verbose output
pytest -v
```

#### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Test fixtures
├── test_llm.py              # LLM class tests
├── test_sampling.py         # Sampling tests
├── test_engine.py           # Engine tests
├── test_layers.py           # Layer tests
└── test_models.py           # Model tests
```

#### Writing Tests
```python
import pytest
from nanovllm import LLM, SamplingParams

class TestLLM:
    def test_basic_generation(self, model_path):
        """Test basic text generation."""
        llm = LLM(model_path)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        
        outputs = llm.generate(["Hello"], sampling_params)
        
        assert len(outputs) == 1
        assert "text" in outputs[0]
        assert "token_ids" in outputs[0]
        assert len(outputs[0]["token_ids"]) <= 10

    def test_batch_generation(self, model_path):
        """Test batch processing."""
        llm = LLM(model_path)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
        
        prompts = ["A", "B", "C"]
        outputs = llm.generate(prompts, sampling_params)
        
        assert len(outputs) == 3
        for output in outputs:
            assert len(output["token_ids"]) <= 5
```

### Documentation

#### Docstring Standards
All public functions and classes must have comprehensive docstrings:

```python
class LLM:
    """High-performance inference engine for large language models.
    
    This class provides a simple interface for text generation with
    optimizations including CUDA graphs, prefix caching, and tensor parallelism.
    
    Args:
        model: Path to HuggingFace model directory
        max_num_batched_tokens: Maximum tokens per batch
        max_num_seqs: Maximum concurrent sequences
        gpu_memory_utilization: GPU memory usage fraction
        
    Example:
        >>> llm = LLM("/path/to/model")
        >>> outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
        >>> print(outputs[0]["text"])
    """
```

#### API Documentation
- Update `docs/api.md` when adding new public APIs
- Include usage examples for all new features
- Document breaking changes clearly

## 🚀 Contributing

### Getting Started

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite**
   ```bash
   pytest
   black nanovllm tests
   isort nanovllm tests
   mypy nanovllm
   ```
7. **Submit a pull request**

### Pull Request Guidelines

#### Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New functionality has tests
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)

#### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Issue Reporting

#### Bug Reports
Include the following information:
- **Environment**: OS, Python version, CUDA version
- **Hardware**: GPU model, memory
- **Reproduction**: Minimal code example
- **Expected vs Actual**: Clear description of issue
- **Logs**: Error messages and stack traces

#### Feature Requests
- **Use Case**: Describe the problem you're solving
- **Proposed Solution**: Your suggested approach
- **Alternatives**: Other solutions you've considered
- **Impact**: Who would benefit from this feature

## 🔧 Extending Nano-vLLM

### Adding New Models

1. **Create model file** in `nanovllm/models/`
2. **Implement required interfaces**:
   ```python
   class YourModel(nn.Module):
       def forward(self, input_ids, positions):
           # Model forward pass
           pass
           
       def compute_logits(self, hidden_states):
           # Compute logits for sampling
           pass
   ```
3. **Add model loading logic** in `utils/loader.py`
4. **Add tests** in `tests/test_models.py`

### Adding New Layers

1. **Create layer file** in `nanovllm/layers/`
2. **Implement tensor parallelism** if needed
3. **Add to model implementations**
4. **Write tests** for the new layer

### Adding New Optimizations

1. **Implement optimization** in appropriate module
2. **Add configuration options** to `Config` class
3. **Update documentation** with usage examples
4. **Add benchmarks** to measure impact

## 🐛 Debugging

### Common Issues

#### CUDA Memory Issues
```python
import torch

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

#### Model Loading Issues
```python
# Debug model loading
from nanovllm.utils.loader import load_model

# Check model files
import os
model_path = "/path/to/model"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Files: {os.listdir(model_path)}")
```

#### Performance Issues
```python
import time
import torch

# Profile generation
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

print(f"Generation time: {end_time - start_time:.2f}s")
print(f"Tokens per second: {total_tokens / (end_time - start_time):.2f}")
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Disable optimizations for debugging
llm = LLM("/path/to/model", enforce_eager=True)
```

## 📊 Performance Profiling

### Memory Profiling
```python
import torch
from nanovllm import LLM

def profile_memory():
    llm = LLM("/path/to/model")
    
    # Monitor memory during generation
    torch.cuda.reset_peak_memory_stats()
    
    outputs = llm.generate(["Test prompt"], SamplingParams(max_tokens=100))
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory usage: {peak_memory:.2f} GB")
```

### Timing Profiling
```python
import time
from nanovllm import LLM

def profile_timing():
    llm = LLM("/path/to/model")
    
    # Warm up
    llm.generate(["Warm up"], SamplingParams(max_tokens=10))
    
    # Profile
    start_time = time.perf_counter()
    outputs = llm.generate(["Test"], SamplingParams(max_tokens=100))
    end_time = time.perf_counter()
    
    print(f"Generation time: {end_time - start_time:.4f}s")
```

## 🔄 Release Process

### Version Management
- **Semantic Versioning**: MAJOR.MINOR.PATCH
- **Changelog**: Update `CHANGELOG.md` for each release
- **Tagging**: Create git tags for releases

### Release Checklist
- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Changelog is updated
- [ ] Version is bumped
- [ ] Release notes are written
- [ ] PyPI package is built and uploaded

### Building Distribution
```bash
# Build package
python -m build

# Upload to PyPI (for maintainers)
twine upload dist/*
```

## 🤝 Community

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first

### Code of Conduct
- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Follow project guidelines

This development guide provides comprehensive information for contributing to Nano-vLLM. For more detailed information about the architecture and API, refer to the [Architecture Overview](./architecture.md) and [API Reference](./api.md). 