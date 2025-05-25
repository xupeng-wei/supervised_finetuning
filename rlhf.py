import torch
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
from transformers import (LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, BitsAndBytesConfig,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tok = LlamaTokenizer.from_pretrained("sft-llama")
tok.pad_token = tok.eos_token

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)

raw_dataset = load_dataset("json", data_files="reasoning_100_examples.jsonl", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
def preprocess(example):
    # return {"query": example["instruction"]}
    return tok("instruction: " + example["instruction"] + "\n answer: ", padding="max_length", truncation=True, max_length=512)

train_dataset = split_dataset["train"].map(preprocess)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset = split_dataset["test"].map(preprocess)
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


# This is the RLHF model, what you are going to train.
policy = AutoModelForCausalLM.from_pretrained("sft-llama",
            quantization_config=bnb_cfg, device_map="auto")
policy = prepare_model_for_kbit_training(policy, use_gradient_checkpointing=True)
policy = get_peft_model(policy, LoraConfig(r=16, target_modules=["q_proj","v_proj"]))

# Same model, but frozen, and is used only to calculate KL divergence
ref_model = AutoModelForCausalLM.from_pretrained("sft-llama",
            quantization_config=bnb_cfg, device_map="auto")

reward_model = AutoModelForSequenceClassification.from_pretrained(
            "rm_llama3b", num_labels=1, torch_dtype=torch.bfloat16, device_map="auto")

# Use the same structure as the reward model
value_model = AutoModelForSequenceClassification.from_pretrained(
            "rm_llama3b", num_labels=1, torch_dtype=torch.bfloat16, device_map="auto")

ppo_cfg = PPOConfig(batch_size=4,
                    local_rollout_forward_batch_size=2,
                    learning_rate=5e-6,
                    num_ppo_epochs=5,
                    temperature=0.7)

ppo_trainer = PPOTrainer(ppo_cfg,
                         model=policy,
                         ref_model=ref_model,
                         reward_model=reward_model,
                         value_model=value_model,
                         processing_class=tok,
                         train_dataset=train_dataset,
                         eval_dataset=train_dataset)

ppo_trainer.train()

policy.save_pretrained("ppo_llama3b")
tok.save_pretrained("ppo_llama3b")