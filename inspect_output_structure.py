from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

def test_output_structure():
    """Inspect the structure of the output from nano-vllm's generate method."""
    print("=== INSPECTING GENERATE OUTPUT STRUCTURE ===")
    
    model_name = "Qwen/Qwen3-0.6B"
    prompt = "Hello"
    
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    
    # Load nano-vllm model
    print("\nLoading nano-vllm model...")
    nano_model = LLM(model_name, device="cpu", dtype="float16")
    
    # Generate with nano-vllm
    print("Generating with nano-vllm model...")
    params = SamplingParams(max_tokens=5, temperature=0.0)
    nano_output = nano_model.generate([prompt], sampling_params=params)
    
    # Inspect output structure
    print("\n=== OUTPUT STRUCTURE ===")
    print(f"Type of output: {type(nano_output)}")
    print(f"Length of output: {len(nano_output)}")
    
    # Inspect first item
    first_item = nano_output[0]
    print(f"\nType of first item: {type(first_item)}")
    
    # Print all attributes and their values
    print("\nAttributes and values:")
    for key, value in first_item.__dict__.items() if hasattr(first_item, "__dict__") else first_item.items():
        print(f"  {key}: {value}")
    
    # If it's a dict, try accessing generated_text
    if isinstance(first_item, dict):
        if "generated_text" in first_item:
            print(f"\nGenerated text: {first_item['generated_text']}")
        elif "text" in first_item:
            print(f"\nText: {first_item['text']}")
        else:
            print("\nCommon keys in dict:", list(first_item.keys())[:5] if len(first_item) > 5 else list(first_item.keys()))
    else:
        print("\nTrying to access attributes...")
        try:
            if hasattr(first_item, "text"):
                print(f"Text: {first_item.text}")
            elif hasattr(first_item, "generated_text"):
                print(f"Generated text: {first_item.generated_text}")
            elif hasattr(first_item, "output"):
                print(f"Output: {first_item.output}")
            else:
                print("No common text attributes found")
        except Exception as e:
            print(f"Error accessing attributes: {e}")

if __name__ == "__main__":
    test_output_structure()
