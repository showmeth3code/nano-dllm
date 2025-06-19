# Installation Guide for tiny-vllm

## Quick Setup

### 1. Create and activate conda environment
```bash
conda create -n tiny-vllm python=3.12 -y
conda activate tiny-vllm
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the project in development mode
```bash
pip install -e .
```

### 4. Build and install Rust components
```bash
# The Rust components are automatically built when installing with maturin
# If you need to rebuild manually:
maturin develop --release
```

## Verification

Test that everything is working:

```bash
# Quick test of all components
python test_imports.py

# Comprehensive integration demo
python demo.py

# Basic example
python example.py

# Test Rust module directly
python -c "import tiny_vllm_py; print('Device:', tiny_vllm_py.get_device())"

# Run tests
python tests/python/epoch02_parity.py
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'tiny_vllm_py'`:
```bash
cd tiny-vllm-py
maturin develop --release
cd ..
```

If you get import errors for nanovllm components:
```bash
pip install -e .
```

If you get import errors for transformers:
```bash
pip install transformers
```

## GPU Support (Optional)

For GPU acceleration on systems with CUDA:

1. Install CUDA drivers and toolkit
2. Uncomment GPU dependencies in `requirements.txt`
3. Install additional dependencies:
   ```bash
   pip install triton flash-attn
   ```
4. Rebuild the project

## Dependencies

### Core Dependencies (required)
- torch>=2.4.0
- transformers>=4.51.0
- xxhash
- nvidia-ml-py3>=7.352.0

### Development Dependencies
- maturin>=1.8.0
- ninja>=1.11.0

### Optional GPU Dependencies
- triton>=3.0.0 (CUDA systems only)
- flash-attn (CUDA systems only)

## Project Structure

```
tiny-vllm/
├── nanovllm/           # Python package (high-level ML operations)
├── tiny-vllm-py/       # Rust extension (low-level system utilities)
├── tiny-vllm-core/     # Core Rust library
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
├── demo.py            # Integration demonstration
└── example.py         # Basic usage example
```
