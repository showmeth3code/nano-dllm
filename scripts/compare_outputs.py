import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

def main():
    # Get model path from environment or use default
    model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen1.5-0.5B")
    print(f"Testing model: {model_name}")
    
    # Load HuggingFace model and tokenizer
    print(f"Loading HuggingFace model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    hf_model.eval()
    
    # Load nano-vllm implementation
    print(f"Loading nano-vllm model...")
    llm = LLM(model_name)
    
    # Set up sampling parameters for nano-vllm
    sampling_params = SamplingParams(
        temperature=0.0,  # Use 0 to make output deterministic
        max_tokens=50,
    )
    
    # Define test prompts
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "The square root of 16 is"
    ]
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\n\n==== PROMPT {i+1}: '{prompt}' ====")
        
        # Generate with HuggingFace
        print("\n--- HuggingFace Output ---")
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.0,  # Use 0 to make output deterministic
                do_sample=False
            )
        hf_text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{hf_text}")
        
        # Generate with nano-vllm
        print("\n--- nano-vllm Output ---")
        outputs = llm.generate([prompt], sampling_params)
        nano_text = outputs[0]["text"]
        print(f"{nano_text}")
        
        # Compare outputs
        print("\n--- Output Comparison ---")
        if hf_text == nano_text:
            print("MATCH: Outputs are identical! ✓")
        else:
            print("MISMATCH: Outputs differ! ✗")
            # Find the point where they start to diverge
            min_len = min(len(hf_text), len(nano_text))
            diverge_idx = next((i for i in range(min_len) if hf_text[i] != nano_text[i]), min_len)
            
            # Show divergence point
            context = 30  # Characters of context before/after
            start_idx = max(0, diverge_idx - context)
            hf_snippet = hf_text[start_idx:diverge_idx + context]
            nano_snippet = nano_text[start_idx:diverge_idx + context]
            
            print(f"Divergence begins at character {diverge_idx}:")
            print(f"HF  : ...{hf_snippet}...")
            print(f"NANO: ...{nano_snippet}...")
    
    print("\n\nComparison complete!")

if __name__ == "__main__":
    main()
