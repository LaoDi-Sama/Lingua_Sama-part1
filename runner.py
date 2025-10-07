#!/usr/bin/env python3
# lora_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========== 配置区 ==========
BASE_MODEL = "./Llama-3.2-3B-Instruct"  # 原始模型路径
LORA_DIR   = "./Lingua_Lora/lingua_lora_fixed_1755095923"  # 你的 LoRA 文件夹
DEVICE     = "cpu"  # 或 "cpu"
MAX_NEW_TOKENS = 100
# ============================

# 1) 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
)

# 3) 加载 LoRA 权重
model = PeftModel.from_pretrained(model, LORA_DIR)
model.to(DEVICE)
model.eval()  # 推理模式

# 4) 推理函数
def generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5) 测试
if __name__ == "__main__":
    prompt = "你好，Lingua！"
    result = generate(prompt)
    print("==== Generated ====")
    print(result)
