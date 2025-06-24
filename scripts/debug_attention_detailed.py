import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import torch.nn.functional as F

sys.path.append(os.getcwd())
from nanovllm.llm import LLM

def test_attention_logic():
    """Test the attention and causal masking logic in both implementations."""
    model_name = "Qwen/Qwen3-0.6B"
    prompt = "Hello, world!"
    device = "cpu"
    
    print(f"=== ATTENTION LOGIC COMPARISON FOR {model_name} ===")
    print(f"Prompt: '{prompt}'")
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    nano_model = LLM(model_name, device=device, dtype="float16")
    
    # Tokenize input
    print("\nTokenizing input...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([id]) for id in input_ids[0].tolist()]}")
    seq_len = input_ids.size(1)
    
    # Extract HuggingFace attention logic
    print("\n=== EXTRACTING HUGGINGFACE ATTENTION LOGIC ===")
    hf_layer0 = hf_model.model.layers[0]
    hf_attn = hf_layer0.self_attn
    
    # Extract attention parameters from HF
    print(f"HF attention type: {type(hf_attn).__name__}")
    print("HF config:")
    for attr in dir(hf_attn):
        if not attr.startswith('_') and not callable(getattr(hf_attn, attr)):
            try:
                print(f"  {attr}: {getattr(hf_attn, attr)}")
            except Exception:
                pass
    
    # Save attention scores and causal mask from HF
    hf_attn_scores = []
    hf_causal_masks = []
    
    def hf_attn_hook(module, input_tensors, output_tensors):
        # Extract the attention scores and masks from HuggingFace attention module
        # Note: This is specific to Qwen3's attention mechanism
        q, k, v = input_tensors
        if hasattr(module, 'last_attn_scores') and module.last_attn_scores is not None:
            # Some HF models save attention scores
            hf_attn_scores.append(module.last_attn_scores.detach().cpu())
        
        # For Qwen3, need to compute the mask
        if seq_len > 1:  # Only create mask for seq_len > 1
            # Create causal mask like HF does
            # From HF code: attention_mask = self._prepare_decoder_attention_mask(...)
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool).triu(diagonal=1)
            hf_causal_masks.append(causal_mask.detach().cpu())
    
    # Register hook on HF model
    hook_handle = hf_attn.register_forward_hook(hf_attn_hook)
    
    print("\nRunning HF forward pass...")
    with torch.no_grad():
        # Move input_ids to the same device as model
        input_ids_hf = input_ids.to(hf_model.device)
        hf_outputs = hf_model(input_ids_hf)
        hf_logits = hf_outputs.logits
    
    # Remove hook
    hook_handle.remove()
    
    print("\nHF attention scores and mask details:")
    if hf_attn_scores:
        print(f"  HF attention scores shape: {hf_attn_scores[0].shape}")
    if hf_causal_masks:
        print(f"  HF causal mask shape: {hf_causal_masks[0].shape}")
        print("  HF causal mask (True means masked):")
        print(f"  {hf_causal_masks[0]}")
        
    # Extract a sample of the attention weights after softmax (for first head)
    if hf_attn_scores:
        sample_attn_scores = hf_attn_scores[0][0, 0]  # batch 0, head 0
        sample_attn_weights = F.softmax(sample_attn_scores, dim=-1)
        print("\nHF sample attention weights after softmax (first head):")
        print(f"{sample_attn_weights}")
    
    # ====== Now test the nano-vllm attention implementation ======
    print("\n\n=== RUNNING NANO-VLLM WITH DEBUG ATTENTION INFO ===")
    
    # Run nano-vllm with the same input
    with torch.no_grad():
        # The LLM class directly takes the prompt
        # Use generate to get a prediction, but we don't actually need the output
        # Just want to trace the attention mechanism
        from nanovllm.sampling_params import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0)
        _ = nano_model.generate([prompt], sampling_params=params)
    
    # Compare outputs
    print("\n=== COMPARING OUTPUTS ===")
    print(f"HF logits shape: {hf_logits.shape}")
    print("HF last token top 5:")
    last_token_logits = hf_logits[0, -1]
    top5 = torch.topk(last_token_logits, 5)
    for i in range(5):
        token_id = top5.indices[i].item()
        token = tokenizer.decode([token_id])
        probs = F.softmax(last_token_logits, dim=-1)
        token_id_tensor = torch.tensor(token_id)
        prob = probs[token_id_tensor].item()
        print(f"  {i+1}. ID={token_id}, Token='{token}', Prob={prob:.4f}")

if __name__ == "__main__":
    test_attention_logic()
