"""
Test script for directly debugging LLMEngine with detailed output
"""

import argparse

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class EngineArgs:
    def __init__(self, model_path, device="cpu", tensor_parallel_size=1):
        self.model = model_path
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = True
        self.max_num_seqs = 256
        self.max_num_batched_tokens = 2048


def print_model_config(config, tokenizer=None):
    """Print key information about the model configuration"""
    print("=== MODEL CONFIGURATION ===")
    print(f"Model path: {config.model_path}")
    # Access HF config for model details
    hf_config = config.hf_config
    print(f"Model type: {getattr(hf_config, 'model_type', 'unknown')}")
    print(f"Vocab size: {getattr(hf_config, 'vocab_size', 'unknown')}")
    print(f"Hidden size: {getattr(hf_config, 'hidden_size', 'unknown')}")
    print(f"Num attention heads: {getattr(hf_config, 'num_attention_heads', 'unknown')}")
    print(f"Num key-value heads: {getattr(hf_config, 'num_key_value_heads', 'unknown')}")
    if tokenizer:
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"EOS token ID: {getattr(hf_config, 'eos_token_id', 'unknown')}")
    print("===============================")


def main():
    parser = argparse.ArgumentParser(description="Debug NanoVLLM LLMEngine")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()
    
    # Initialize engine arguments
    engine_args = EngineArgs(args.model, args.device)
    
    print(f"Testing LLMEngine with {args.model} on {args.device} device")
    print("Initializing LLMEngine...")
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer
    
    # Print model configuration
    print_model_config(engine.config, tokenizer)
    
    # Create sampling parameters
    prompt = args.prompt
    print(f"Prompt: '{prompt}'")
    
    params = SamplingParams(
        temperature=args.temp, 
        max_tokens=args.max_tokens,
        ignore_eos=True
    )
    
    # Tokenize to show token IDs
    ids = tokenizer.encode(prompt)
    print(f"Tokenized: {ids}")
    print()
    
    # Generate output
    print("Generating text...")
    outputs = engine.generate([prompt], params)
    
    # Process and print results
    print("\nGeneration results:")
    for i, output in enumerate(outputs):
        print(f"Output {i+1}:")
        print(f"  Text: {repr(output['text'])}")
        
        # Parse and print token IDs
        token_ids = output["token_ids"]
        print(f"  Token IDs: {token_ids}")
        
        # Print individual tokens
        print("  Individual tokens:")
        for j, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id])
            print(f"    {j}: {token_id} → {repr(token_text)}")


if __name__ == "__main__":
    main()
