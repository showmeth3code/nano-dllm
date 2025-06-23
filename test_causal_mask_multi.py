import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
import logging

# Set up logging to see debug information
logging.basicConfig(level=logging.DEBUG)

# Create a test to verify causal mask behavior with multiple tokens
def test_causal_mask():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load both our model and HF model for comparison
    print("Loading nano-vllm model...")
    model = LLM("Qwen/Qwen3-0.6B", device=device)
    
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float16, trust_remote_code=True)
    hf_model = hf_model.to(device)
    hf_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Create a simple input with multiple tokens
    input_text = "This is a test for causal mask"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"Input ID shape: {input_ids.shape}")
    print(f"Sequence length: {input_ids.shape[1]}")
    print(f"Input tokens: {tokenizer.decode(input_ids[0])}")
    
    # Create a sampling params object
    sampling_params = SamplingParams(
        temperature=0.0,  # Use greedy decoding
        max_tokens=0  # Don't generate any new tokens, just get logits for input
    )
    
    # Process through nano-vllm to check causal mask behavior
    print("\n--- Testing nano-vllm model with multiple tokens ---")
    outputs = model.generate([input_text], sampling_params)
    print(f"Generated output: {outputs[0]['text']}")
    
    # Test with a single token
    print("\n--- Testing nano-vllm model with single token ---")
    single_token_text = tokenizer.decode([input_ids[0, 0]])
    single_outputs = model.generate([single_token_text], sampling_params)
    
    print(f"Single token text: '{single_token_text}'")
    print(f"Single output: {single_outputs[0]['text']}")
    
    # Compare with HuggingFace model
    print("\n--- Testing HuggingFace model with multiple tokens ---")
    with torch.no_grad():
        hf_outputs = hf_model(input_ids).logits
    
    print(f"HF output shape: {hf_outputs.shape}")
    
    # Now let's also test generation with a longer sequence
    print("\n--- Testing generation with causal mask ---")
    
    # Sample text that will require causal masking
    generation_text = "Answer the following question: What is the capital of France"
    
    # Create sampling parameters for generation
    gen_params = SamplingParams(temperature=0.0, max_tokens=10)  # Generate 10 tokens
    
    # Generate with nano-vllm
    print("Generating with nano-vllm:")
    nano_gen = model.generate([generation_text], gen_params)
    print(f"Input: '{generation_text}'")
    print(f"Output: '{nano_gen[0]['text']}'")
    
    # Generate with HuggingFace for comparison
    print("\nGenerating with HuggingFace:")
    gen_ids = tokenizer.encode(generation_text, return_tensors="pt").to(device)
    with torch.no_grad():
        hf_gen = hf_model.generate(
            gen_ids,
            max_new_tokens=10,
            do_sample=False,  # Use greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
    hf_gen_text = tokenizer.decode(hf_gen[0], skip_special_tokens=True)
    print(f"Input: '{generation_text}'")
    print(f"Output: '{hf_gen_text}'")
    
    # Compare outputs
    print("\n--- Comparison ---")
    nano_output_only = nano_gen[0]['text'].replace(generation_text, "").strip()
    hf_output_only = hf_gen_text.replace(generation_text, "").strip()
    
    print(f"nano-vllm generated: '{nano_output_only}'")
    print(f"HuggingFace generated: '{hf_output_only}'")
    print(f"Match: {'✅' if nano_output_only == hf_output_only else '❌'}")
    
    return "Test complete"

if __name__ == "__main__":
    result = test_causal_mask()
    print(f"\n=== {result} ===")
