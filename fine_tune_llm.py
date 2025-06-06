import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# -----------------------------
# CONFIGURATION
# -----------------------------
EXCEL_PATH = "your_dataset.xlsx"  # 👈 Update this
MODEL_PATH = "path_to_gemma_12b"  # 👈 Local path to Gemma 3 12B
OUTPUT_DIR = "./gemma3_12b_lora"
TEXT_COLUMN = "text"
EPOCHS = 3
BATCH_SIZE = 1
MAX_LENGTH = 512

# -----------------------------
# LOAD DATASET
# -----------------------------
print("📖 Loading dataset...")
df = pd.read_excel(EXCEL_PATH)
df = df[[TEXT_COLUMN]].dropna().rename(columns={TEXT_COLUMN: "text"})

dataset = Dataset.from_pandas(df)

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
print("🔧 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

# -----------------------------
# TOKENIZATION
# -----------------------------
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

print("✂️ Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True)

# -----------------------------
# APPLY PEFT (LoRA)
# -----------------------------
print("🪄 Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# -----------------------------
# TRAINING SETUP
# -----------------------------
print("🚀 Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
print("✅ Training complete!")