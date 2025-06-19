#!/bin/bash

echo "=== Setting up Tiny-vLLM ==="

# Check if conda environment exists
if ! conda env list | grep -q "tiny-vllm"; then
    echo "Creating conda environment..."
    conda create -n tiny-vllm python=3.12 -y
fi

# Activate conda environment
echo "1. Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tiny-vllm

# Install Python dependencies
echo "2. Installing Python dependencies..."
pip install -r requirements.txt

# Build Rust extension
echo "3. Building Rust extension..."
cd tiny-vllm-py
maturin develop --release
cd ..

# Install project in development mode
echo "4. Installing project in development mode..."
pip install -e .

# Test imports
echo "5. Testing imports..."
python -c "
try:
    import tiny_vllm_py
    print('✓ tiny_vllm_py imported successfully')
    print(f'  Device: {tiny_vllm_py.get_device()}')
except ImportError as e:
    print(f'✗ Failed to import tiny_vllm_py: {e}')

try:
    from nanovllm.sampling_params import SamplingParams
    print('✓ nanovllm imported successfully')
except ImportError as e:
    print(f'✗ Failed to import nanovllm: {e}')

try:
    from transformers import AutoTokenizer
    print('✓ transformers imported successfully')
except ImportError as e:
    print(f'✗ Failed to import transformers: {e}')
"

echo ""
echo "=== Setup complete! ==="
echo "To test the setup, run:"
echo "  conda activate tiny-vllm"
echo "  python demo.py"
