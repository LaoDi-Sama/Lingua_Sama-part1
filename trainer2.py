#!/usr/bin/env python3
# trainer2.py
"""
LoRA 微调脚本（ROCm 6.3 + GPTQ-4bit）
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

########################################
# 1. 路径配置（改成你的）
MODEL_PATH = "./Llama-3.2-3B-Instruct"   # 量化后的底座
DATA_PATH  = "./TrainingDatas/Trainning_data.jsonl"                # 毒舌对话
OUTPUT_DIR = "./Lingua_Lora/lingua_lora"
########################################

########################################
# 2. 加载 GPTQ 底座 + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    device_map="auto",
    use_triton=False,         # ROCm 先关掉 triton
    torch_dtype=torch.float16
)
########################################

########################################
# 3. LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 小模型够用
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
########################################

########################################
# 4. 加载毒舌数据
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

# 模板化：Llama-3.2 官方 chat 模板
def template(example):
    text = (
        f"<|start_header_id|>user<|end_header_id|>\n{example['prompt']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{example['response']}<|eot_id|>"
    )
    return tokenizer(text, truncation=True, max_length=512)

dataset = dataset.map(template, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
########################################

########################################
# 5. 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_32bit",
    report_to=None
)
########################################

########################################
# 6. 启动训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
########################################

print("✅ LoRA 训练完成，权重保存在", OUTPUT_DIR)