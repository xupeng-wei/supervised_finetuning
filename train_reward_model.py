import torch, json
from datasets import load_dataset
from transformers import (LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

BASE = "openlm-research/open_llama_3b"

tok = LlamaTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)

rm = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=1, quantization_config=bnb_cfg, device_map="auto")
rm = prepare_model_for_kbit_training(rm, use_gradient_checkpointing=True)
rm = get_peft_model(rm, LoraConfig(r=16, lora_alpha=32,
                                   target_modules=["q_proj","v_proj"]))

# TODO: find another RLHF dataset for our purposes
ds = load_dataset("Anthropic/hh-rlhf", split="train")
def to_rank_pairs(batch):
    texts = []
    labels = []
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        texts.append(chosen)
        labels.append(1.0)
        texts.append(rejected)
        labels.append(0.0)

    tokenized = tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    tokenized["labels"] = labels
    return tokenized

# train_ds = ds.map(to_rank_pairs, batched=True, remove_columns=ds.column_names)
train_ds = ds.select(range(1000)).map(to_rank_pairs, batched=True, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="rm_llama3b",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    bf16=True,
    logging_steps=50,
    save_strategy="no"
)

Trainer(model=rm, args=args, train_dataset=train_ds).train()
rm.save_pretrained("rm_llama3b")
tok.save_pretrained("rm_llama3b")