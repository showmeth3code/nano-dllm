from transformers import AutoModelForCausalLM, AutoTokenizer

# Change this to any HuggingFace-compatible model you want
def download_model(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Downloading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print("Download complete.")

if __name__ == "__main__":
    download_model()
