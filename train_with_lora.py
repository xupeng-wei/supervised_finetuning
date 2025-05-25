
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Load dataset
raw_dataset = load_dataset("json", data_files="reasoning_100_examples.jsonl", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Tokenizer
model_name = "openlm-research/open_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Prevents padding errors

# Preprocessing
def preprocess(example):
    instruction = example["instruction"]
    answer = example["answer"]
    explanation = example.get("explanation", "")
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{answer}\nExplanation: {explanation}\n"
    tokens = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens

train_dataset = train_dataset.map(preprocess, batched=False)
eval_dataset = eval_dataset.map(preprocess, batched=False)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# BitsAndBytes config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./sft-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train
trainer.train()

# Save
trainer.save_model("./sft-llama")
tokenizer.save_pretrained("./sft-llama")
