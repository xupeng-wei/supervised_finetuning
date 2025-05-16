import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load data
raw_dataset = load_dataset("json", data_files="reasoning_100_examples.jsonl", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load tokenizer
model_name = "openlm-research/open_llama_3b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Important!

# Preprocessing function
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

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, batched=False)
eval_dataset = eval_dataset.map(preprocess, batched=False)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training args
training_args = TrainingArguments(
    output_dir="./sft-llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
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

# Save model
trainer.save_model("./my_sft_model")
tokenizer.save_pretrained("./my_sft_model")

# Run
# accelerate config
# Then 
# accelerate launch train_script.py
# Or if using multiple GPUs
# torchrun --nproc_per_node=2 train_script.py
