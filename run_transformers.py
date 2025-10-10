from transformers import pipeline

try:
    model_id = "openai/gpt-oss-20b"
    pipe = pipeline("text-generation", model=model_id, torch_dtype="auto", device_map="auto")
except Exception as e:
    print("⚠️ Could not load gpt-oss-20b, falling back to gpt2.")
    print("Error:", e)
    model_id = "gpt2"
    pipe = pipeline("text-generation", model=model_id)

messages = [{"role": "user", "content": "Explain quantum mechanics clearly and concisely."}]
outputs = pipe(messages, max_new_tokens=100)
print(outputs[0]["generated_text"])
