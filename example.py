import os

# Robust imports with error handling
def safe_import():
    imports = {}
    
    try:
        from nanovllm import LLM, SamplingParams
        imports['LLM'] = LLM
        imports['SamplingParams'] = SamplingParams
        print("✓ Successfully imported nanovllm")
    except ImportError as e:
        print(f"⚠ Could not import nanovllm: {e}")
        imports['LLM'] = None
        imports['SamplingParams'] = None
    
    try:
        from transformers import AutoTokenizer
        imports['AutoTokenizer'] = AutoTokenizer
        print("✓ Successfully imported transformers")
    except ImportError as e:
        print(f"⚠ Could not import transformers: {e}")
        imports['AutoTokenizer'] = None
    
    return imports


def main():
    # Import with error handling
    imports = safe_import()
    LLM = imports.get('LLM')
    SamplingParams = imports.get('SamplingParams')
    AutoTokenizer = imports.get('AutoTokenizer')
    
    print()
    
    if not AutoTokenizer:
        print("❌ AutoTokenizer not available - cannot run example")
        print("Fix: pip install transformers")
        return
    
    # Use a smaller model that can be downloaded from Hugging Face
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # For now, just test the tokenizer - the full LLM might need CUDA
    print(f"Successfully loaded tokenizer for {model_name}")
    
    # Test tokenization
    test_prompt = "Hello, how are you?"
    tokens = tokenizer.encode(test_prompt)
    print(f"Tokenized '{test_prompt}' to: {tokens}")
    
    if LLM and SamplingParams:
        print("✓ LLM and SamplingParams available for full functionality")
        # Uncomment the following lines when you have a proper model setup
        # llm = LLM(model_name, enforce_eager=True, tensor_parallel_size=1)
        # sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
        # prompts = ["introduce yourself", "list all prime numbers within 100"]
        # outputs = llm.generate(prompts, sampling_params)
        # for prompt, output in zip(prompts, outputs):
        #     print(f"Prompt: {prompt!r}")
        #     print(f"Completion: {output['text']!r}")
    else:
        print("⚠ LLM components not available - only testing tokenization")
        print("Fix: pip install -e . to install nanovllm")
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
