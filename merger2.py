
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import tkinter as tk
from tkinter import filedialog
import time
import os

# 时间戳，用于区分输出目录
current_time = time.strftime("%Y%m%d-%H%M%S")

# 选择 LoRA 文件夹
root1 = tk.Tk()
root1.withdraw()
file_ds = filedialog.askdirectory(title="选择LoRA所在文件夹，选好了点第一个")
print("你选择了：", file_ds)

# 基础模型路径（改成你自己的路径）
base_model_path = "/home/laodi/LLM_Lora_Trainer/Llama-3.2-3B-Instruct"

# LoRA 权重路径
lora_model_path = file_ds

# 输出路径
output_path = f"/home/laodi/LLM_Lora_Trainer/merged_model/Lingua{current_time}"
os.makedirs(output_path, exist_ok=True)

# 加载基础模型
print("加载基础模型...")
model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# 加载 LoRA 并合并
print("合并LoRA权重...")
model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.float16)
model = model.merge_and_unload()

# 保存合并后的模型
print("保存合并后的模型...")
model.save_pretrained(output_path, safe_serialization=True)

# 保存 tokenizer（保证和模型匹配）
print("保存tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.save_pretrained(output_path)

print(f"✅ 模型已保存到: {output_path}")
print("接下来你可以运行：")
print("python convert-hf-to-gguf.py {output_path} --outfile merged-f16.gguf")
print("然后再用 llama-quantize 进行量化。")

