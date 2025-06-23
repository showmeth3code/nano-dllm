import time
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.utils.loader import load_model
from transformers import AutoTokenizer
from nanovllm.sampling_params import SamplingParams


models = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B",
]

messages = [
    {"role": "user", "content": "Who are you?"},
]

for model_name in models:
    print(f"Testing model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model configuration
    config = Config(model_path=model_name)
    
    # Initialize LLMEngine
    engine = LLMEngine(config=config, engine_args={}, tokenizer=tokenizer)
    
    # Generate text
    prompt = messages[0]["content"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100, ignore_eos=False)  # Example parameters
    engine.add_request(prompt, sampling_params)
    
    # Simulate output (replace with actual engine response handling)
    output = [{"generated_text": messages}]
    print(output)
    print("=" * 40)