from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/Users/igor.okulist/models/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
