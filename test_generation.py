import os
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

def main():
    model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen1.5-0.5B")
    print(f"Loading model {model_name}")

    # Initialize the LLM
    llm = LLM(model_name)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )
    
    # Define prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about programming.",
        "What are the key features of Python?"
    ]
    
    # Generate text
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        print(f"\n=== Prompt: {prompts[i]} ===")
        print(f"Generated text: {output['text']}")
        
    print("Generation complete!")

if __name__ == "__main__":
    main()
