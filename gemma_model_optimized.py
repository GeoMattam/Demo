from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

# === CONFIG ===
model_id = "google/gemma-3-12b-it"
use_float16 = True       # Set False to use bfloat16
use_compile = True       # Set False if torch.compile causes issues
max_tokens = 64          # Speed up by limiting generation length

# === Load model ===
dtype = torch.float16 if use_float16 else torch.bfloat16

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto"
).eval()

# Optional: Compile the model (for PyTorch 2+)
if use_compile:
    try:
        model = torch.compile(model)
    except Exception as e:
        print("⚠️ torch.compile failed. Proceeding without it.\n", e)

# === Load processor ===
processor = AutoProcessor.from_pretrained(model_id)

# === Define chat messages ===
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Explain Quantum Computing in simple terms."}]
    }
]

# === Process input ===
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
)

# Move inputs to model device with correct dtype
inputs = {k: v.to(model.device, dtype=dtype) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

# === Run inference ===
with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=5)  # Warm-up call (optional)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    output = output[0][input_len:]  # Remove prompt tokens from output

# === Decode output ===
decoded = processor.decode(output, skip_special_tokens=True)
print("\n🧠 Model Output:\n", decoded)
