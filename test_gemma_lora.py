import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# PATHS (update as needed)
# -----------------------------
BASE_MODEL_PATH = "path_to_gemma_12b"       # 👈 Your local Gemma 12B model
LORA_ADAPTER_PATH = "./gemma3_12b_lora"     # 👈 Output dir from training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD BASE MODEL + LoRA
# -----------------------------
print("🚀 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

model.eval()

# -----------------------------
# RUN INFERENCE
# -----------------------------
while True:
    prompt = input("\n📝 Enter your prompt (or 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n🧠 Response:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))