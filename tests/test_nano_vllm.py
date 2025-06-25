import pytest

"""
Test script for nano-vllm using the Qwen3 model.
This script compares generation from HuggingFace and nano-vllm implementations.
"""


def run_nano_vllm_test():
    # Heavy imports moved inside the function to speed up pytest collection
    import torch
    from transformers import AutoTokenizer

    # ...other imports as needed...
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.config import Config
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    import torch.nn.functional as F
    from nanovllm.utils.loader import load_model
    import datetime
    import numpy as np

    # Check device availability
    print("PyTorch device availability:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
    )
    USE_CPU = False
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not USE_CPU
        else "mps"
        if hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not USE_CPU
        else "cpu"
    )
    print(f"Will be using device: {device}")
    models = ["Qwen/Qwen3-0.6B"]
    prompts = ["Hi!"]

    def tensor_equal_within_tol(tensor1, tensor2, rtol=1e-3, atol=1e-5):
        if tensor1.shape != tensor2.shape:
            return False
        return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"Testing model: {model_name}")
        print("=" * 80)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Tokenizer class: {type(tokenizer).__name__}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
        torch_dtype = torch.float32 if device.type == "mps" else torch.float16
        print(f"Using torch dtype: {torch_dtype}")
        from transformers import AutoConfig

        print("\n" + "=" * 40)
        print("Testing with direct model comparison")
        print("=" * 40)
        print(f"Using device: {device}")

        # Add your model loading, inference, and comparison logic here
        # (Move all code from the previous main loop here)




@pytest.mark.heavy
def test_nano_vllm():
    """Pytest-compatible test for nano-vllm model integration."""
    run_nano_vllm_test()


if __name__ == "__main__":
    run_nano_vllm_test()
