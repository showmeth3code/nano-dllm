from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_official_hf():
    print("=== Testing Official HuggingFace Pipeline for Qwen3-0.6B ===\n")
    
    model_name = "Qwen/Qwen3-0.6B"
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Use MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    print("✓ Model and tokenizer loaded successfully")
    
    # Test prompts
    prompts = [
        "Write a short story about a robot learning to paint:",
        "Explain quantum computing in simple terms:",
        "Write a poem about the ocean:",
        "What are the benefits of renewable energy?",
        "Tell me a joke:"
    ]
    
    print(f"\n=== Testing {len(prompts)} prompts ===\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"--- Prompt {i} ---")
        print(f"Input: {prompt}")
        
        # Format as chat message using the official template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Use thinking mode
        )
        
        # Tokenize and generate
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate with official best practices
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the generated part (not the input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if present
        try:
            # Find </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            print(f"Thinking: {thinking_content}")
            print(f"Output: {content}")
        except ValueError:
            # No thinking content found
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            print(f"Output: {content}")
        
        print(f"Raw tokens: {output_ids[:20]}...")
        print()

if __name__ == "__main__":
    test_official_hf() 