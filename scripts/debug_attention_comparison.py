import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm.llm import LLM

def create_hf_causal_mask(size, device):
    """Create a causal mask for attention in HF style."""
    mask = torch.ones((size, size), device=device, dtype=torch.bool).triu(diagonal=1)
    # In HF, True means masked positions, where we'll set -inf
    return mask

def create_nano_causal_mask(size, device):
    """Create a causal mask for attention as in nano-vllm."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    # Convert to boolean where True means masked positions
    return mask.bool()

def main():
    print("=== ATTENTION IMPLEMENTATION COMPARISON ===")
    
    # Load models
    print("Loading tokenizer and models...")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nano_model = LLM(model_name, device='cpu', dtype="float16")
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create a simple prompt
    prompt = "Hi!"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Compare causal masks
    seq_len = input_ids.size(1)
    print(f"\n=== COMPARING CAUSAL MASKS (seq_len={seq_len}) ===")
    
    # Create HF-style causal mask
    hf_mask = create_hf_causal_mask(seq_len, "cpu")
    print("HF causal mask (True means masked):")
    print(hf_mask)
    
    # Create nano-vllm causal mask
    nano_mask = create_nano_causal_mask(seq_len, "cpu")
    print("nano-vllm causal mask (True means masked):")
    print(nano_mask)
    
    # Check if masks are equivalent
    if torch.equal(hf_mask, nano_mask):
        print("✅ MASKS MATCH! The masking logic is identical.")
    else:
        print("❌ MASKS DO NOT MATCH! This is likely causing the issue.")
    
    # Extract attention modules
    print("\n=== EXAMINING ATTENTION IMPLEMENTATIONS ===")
    
    # Extract first attention layer from HF model
    hf_attn = hf_model.model.layers[0].self_attn
    print(f"HF attention module: {hf_attn.__class__.__name__}")
    
    # Extract first attention layer from nano model
    nano_attn = nano_model.model.layers[0].self_attn
    print(f"nano-vllm attention module: {nano_attn.__class__.__name__}")
    
    # Run input through both models with hooks to capture attention scores
    hf_attn_scores = []
    
    def hf_hook(module, input, output):
        """Hook to capture attention scores in HF model."""
        # Get attention scores
        if hasattr(module, "last_attn_scores"):
            hf_attn_scores.append(module.last_attn_scores.detach().cpu())
    
    # Register hooks for HF
    hooks = []
    for layer in hf_model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(hf_hook))
    
    # Run forward pass on HF model to get attention scores
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get attention scores from first layer in HF model
    if hf_attn_scores:
        print("\n=== ATTENTION SCORES FROM FIRST LAYER ===")
        first_layer_scores = hf_attn_scores[0]
        print(f"HF attention scores shape: {first_layer_scores.shape}")
        
        # Print first few attention scores for analysis
        print("HF attention scores sample (first 2x2 block from first head):")
        print(first_layer_scores[0, 0, :2, :2])
        
        # Check if the scores have been properly masked
        if torch.isinf(first_layer_scores[0, 0, 0, 1]):
            print("✅ HF implementation properly masks future tokens (-inf value detected)")
        else:
            print("❓ HF implementation doesn't seem to mask future tokens (no -inf values found)")
            # Show the attention weights after softmax
            attn_weights = torch.nn.functional.softmax(first_layer_scores, dim=-1)
            print("HF attention weights after softmax (first 2x2 from first head):")
            print(attn_weights[0, 0, :2, :2])
    
    print("\n=== RUNNING NANO-VLLM MODEL WITH DEBUG ===")
    
    # Modify attention.py to print information about the causal mask
    # By default, nano_model should already include our debugging code
    # from previous script
    
    # Run forward pass on nano model
    with torch.no_grad():
        nano_outputs = nano_model(input_ids)
    
    # Compare the model outputs
    print("\n=== COMPARING MODEL OUTPUTS ===")
    hf_logits = hf_outputs.logits[:, -1]  # Last token logits
    nano_logits = nano_outputs[:, -1]
    
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"nano-vllm logits shape: {nano_logits.shape}")
    
    # Generate from both models and compare
    print("\n=== GENERATING FROM BOTH MODELS ===")
    
    # HF generate
    hf_gen_ids = hf_model.generate(
        input_ids, 
        max_new_tokens=5, 
        do_sample=False
    )
    
    hf_generated = tokenizer.decode(hf_gen_ids[0])
    print(f"HF generated: {hf_generated}")
    print(f"HF generated token IDs: {hf_gen_ids.tolist()}")
    
    # nano-vllm generate
    from nanovllm.sampling_params import SamplingParams
    
    params = SamplingParams(
        max_tokens=5,
        temperature=0.0  # equivalent to do_sample=False
    )
    
    nano_generated = nano_model.generate([prompt], sampling_params=params)[0].outputs[0].text
    print(f"nano-vllm generated: {prompt + nano_generated}")
    
    # Compare top tokens
    hf_top5 = torch.topk(hf_logits, 5)
    nano_top5 = torch.topk(nano_logits, 5)
    
    print("\n=== TOP 5 TOKENS FROM EACH MODEL ===")
    print("HF top tokens:")
    for i in range(5):
        token_id = hf_top5.indices[0, i].item()
        token = tokenizer.decode([token_id])
        prob = torch.nn.functional.softmax(hf_logits, dim=-1)[0, token_id].item()
        print(f"{i+1}. Token ID: {token_id}, Token: '{token}', Probability: {prob:.4f}")
    
    print("\nnano-vllm top tokens:")
    for i in range(5):
        token_id = nano_top5.indices[0, i].item()
        token = tokenizer.decode([token_id])
        prob = torch.nn.functional.softmax(nano_logits, dim=-1)[0, token_id].item()
        print(f"{i+1}. Token ID: {token_id}, Token: '{token}', Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
