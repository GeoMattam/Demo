from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
import torch

# Paths
base_model_path = "path_to_gemma_12b"
lora_adapter_path = "gemma3_12b_lora"
merged_model_output_path = "merged_gemma_lora_model"

# Load base model + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()  # returns full model with LoRA applied

# Save the full merged model
merged_model.save_pretrained(merged_model_output_path)
tokenizer.save_pretrained(merged_model_output_path)

print(f"✅ Merged model saved to {merged_model_output_path}")

'''

A conversion script: convert.py or transformers-to-gguf.py (found in llama.cpp/scripts)

python3 convert.py \
  --model-path merged_gemma_lora_model \
  --outfile ggml-model-f16.gguf \
  --outtype f16  # or q4_0 / q5_1 / q8_0 for quantized formats

python3 transformers-to-gguf.py \
  --model-dir ./merged_gemma_lora_model \
  --outfile ./gemma-lora.q4_0.gguf \
  --outtype q4_0   # Options: f16, q4_0, q4_k_m, q5_1, q8_0 etc.

Modelfile

FROM ./ggml-model-f16.gguf
NAME gemma-lora

ollama create gemma-lora -f Modelfile
ollama run gemma-lora
'''