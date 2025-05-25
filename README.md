# Supervised Fine-Tuning of LLaMA-3B with Reasoning Examples

This repository provides code and resources for fine-tuning the LLaMA-3B language model using a small, curated dataset of 100 reasoning examples. It offers two training approaches:

- Standard Fine-Tuning: Directly fine-tunes the full model weights.

- LoRA-Based Fine-Tuning: Applies Low-Rank Adaptation (LoRA) for parameter-efficient training.

## 📂 Dataset

- File: `reasoning_100_examples.jsonl`

- Format: JSON Lines, each containing a prompt-response pair.

- Purpose: Designed to enhance the model's reasoning capabilities through supervised fine-tuning.

## 🛠️ Training Scripts

1. `train_script.py` – Full Model Fine-Tuning
Description: Performs standard supervised fine-tuning by updating all model parameters.

Requirements: High computational resources; may encounter out-of-memory (OOM) issues on limited hardware.

2. `train_with_lora.py` – LoRA-Based Fine-Tuning
Description: Implements parameter-efficient fine-tuning using LoRA, which introduces trainable rank decomposition matrices into each layer of the Transformer architecture.

3. `train_reward_model.py` - Reward Model Preparation

4. `rlhf.py` - RLHF
Description: Load the SFT model, RLHF with the prepared reward model.

Advantages:

- Significantly reduces memory usage.

- Enables fine-tuning on consumer-grade GPUs.

- Faster training times compared to full fine-tuning.