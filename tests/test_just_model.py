import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "Who are you?"

def run_just_model_test():
    print("\n=== LOADING HF MODEL (REFERENCE) ===")
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
    hf_model.eval()
    input_ids = hf_tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"Input ids: {input_ids[0].tolist()}")
    print(f"Decoded: {hf_tokenizer.decode(input_ids[0])}")
    print("\n=== RUNNING HF MODEL INFERENCE ===")
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        hf_output_ids = outputs.sequences[0].tolist()
        hf_output_text = hf_tokenizer.decode(hf_output_ids[input_ids.shape[1]:])
    print(f"HF output ids: {hf_output_ids[input_ids.shape[1]:20]}")
    print(f"HF output text: {repr(hf_output_text)}")
    print("\n=== LOADING NANO_VLLM MODEL ===")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    nano_model = Qwen3ForCausalLM(config)
    load_model(nano_model, MODEL_NAME)
    nano_model.eval()
    print("\n=== RUNNING NANO_VLLM MODEL INFERENCE ===")
    input_tensor = torch.tensor(input_ids[0].tolist())
    positions = torch.arange(len(input_tensor))
    with torch.no_grad():
        print(f"Running forward pass for input shape: {input_tensor.shape}")
        logits = nano_model(input_tensor.unsqueeze(0), positions.unsqueeze(0))
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
    print(f"Predicted next token id: {next_token_id}")
    print(f"Predicted token: {repr(hf_tokenizer.decode([next_token_id]))}")
    expected_next_token = hf_output_ids[len(input_ids[0])]
    print(f"Expected next token id: {expected_next_token}")
    print(f"Expected token: {repr(hf_tokenizer.decode([expected_next_token]))}")
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        hf_next_token_logits = hf_logits[0, -1, :]
        hf_top_k = torch.topk(hf_next_token_logits, 5)
        hf_top_ids = hf_top_k.indices.tolist()
        nano_top_k = torch.topk(next_token_logits, 5)
        nano_top_ids = nano_top_k.indices.tolist()
    print("\n=== TOP-5 TOKEN COMPARISON ===")
    print(f"HF top-5: {hf_top_ids}, tokens: {[hf_tokenizer.decode([i]) for i in hf_top_ids]}")
    print(f"NANO top-5: {nano_top_ids}, tokens: {[hf_tokenizer.decode([i]) for i in nano_top_ids]}")
    print("\n=== GENERATING SEQUENCE WITH NANO_VLLM MODEL ===")
    curr_ids = input_ids[0].tolist()
    for i in range(10):
        input_tensor = torch.tensor(curr_ids)
        positions = torch.arange(len(input_tensor))
        with torch.no_grad():
            logits = nano_model(input_tensor.unsqueeze(0), positions.unsqueeze(0))
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
        curr_ids.append(next_token_id)
        print(f"Token {i+1}: {next_token_id} -> {repr(hf_tokenizer.decode([next_token_id]))}")
    final_output = hf_tokenizer.decode(curr_ids)
    print("\nFinal output: {}".format(repr(final_output)))



def test_just_model():
    """Pytest-compatible test for just the model component."""
    run_just_model_test()

if __name__ == "__main__":
    run_just_model_test()
