"""最小复现 SFT segfault 的调试脚本。"""

import sys

sys.path.insert(0, "src")

print("step 0: imports...")
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from echo.data import load_sft_data

print("step 1: model loading...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
print("step 2: prepare kbit...")
model = prepare_model_for_kbit_training(model)

print("step 3: lora...")
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)

print("step 4: tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

print("step 5: data...")
ds = load_sft_data(Path("data/sft/train.jsonl"))
ds = ds.select(range(50))

print("step 6: TrainingArguments...")
training_args = TrainingArguments(
    output_dir="checkpoints/test",
    max_steps=2,
    per_device_train_batch_size=1,
    logging_steps=1,
    report_to="none",
    save_strategy="no",
    bf16=True,
)

print("step 7: SFTTrainer init...")
sys.stdout.flush()
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tok,
    max_seq_length=512,
)

print("step 8: train...")
sys.stdout.flush()
trainer.train()

print("DONE!")
