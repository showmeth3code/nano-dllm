import sys
import os

sys.path.append(os.getcwd())
from nanovllm.llm import LLM 
from nanovllm.sampling_params import SamplingParams

def test_causal_mask_fix():
    """Test if the causal mask fix helps generate better results."""
    print("=== TESTING CAUSAL MASK FIX ===")
    
    model_name = "Qwen/Qwen3-0.6B"
    device = "cpu"
    test_prompts = [
        "Hello",
        "Once upon a time",
        "Write a haiku about",
        "The capital of France is"
    ]
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = LLM(model_name, device=device, dtype="float16")
    
    # Generate for each prompt
    print("\nGenerating responses:")
    params = SamplingParams(max_tokens=15, temperature=0.0)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        
        # Generate
        outputs = model.generate([prompt], sampling_params=params)
        generated_text = outputs[0]["text"]
        print(f"Generated: '{prompt}{generated_text}'")
    
    print("\n=== GENERATION TEST COMPLETE ===")

if __name__ == "__main__":
    test_causal_mask_fix()
