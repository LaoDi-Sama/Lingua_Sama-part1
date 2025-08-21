#!/usr/bin/env python3
# trainer4_fixed.py
"""
LoRA 微调脚本（ROCm 6.3 + RX 6600 修正版）
- 避免 HIP invalid device function (dtype 转换 Bug)
- 采用 FP16 + device_map="auto" + CPU/disk offload
- 默认使用【方案1】：在 CPU 上添加 LoRA，再搬回 GPU
- 备用【方案2】：强制 LoRA dtype 为 fp16，避免 PEFT 的 dtype 转换
- 每 20 step 打印一次显存占用
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import time

import tkinter as tk
from tkinter import filedialog

root1 = tk.Tk()
root1.withdraw()
current_time = time.strftime("%Y%m%d-%H%M%S")
file_ds = filedialog.askopenfilename(title="选择Dataset，选好了点第一个")
print("你选择了：", file_ds)


# ========= 配置区 =========
MODEL_PATH = "./Llama-3.2-3B-Instruct"
# DATA_PATH  = "./TrainingDatas/Trainning_data.jsonl"
DATA_PATH = file_ds
timestamp = int(time.time())  # 时间戳
# OUTPUT_DIR = "./Lingua_Lora/lingua_lora_fixed"
OUTPUT_DIR = f"./Lingua_Lora/lingua_lora_fixed_{current_time}"
OFFLOAD_DIR = "./offload_dir"
GPU_MEM_LIMIT = "5GiB"   # RX6600 建议 4.5~5.5GiB，爆显存就再收紧
CPU_MEM_LIMIT = "28GiB"
LOG_VRAM_EVERY_STEPS = 20
USE_PLAN_1 = True        # True: 方案1（CPU上加LoRA再搬回GPU）；False: 方案2（强制fp16 dtype）
# ========================

# 建议设置环境变量避免 ROCm gfx bug（确保在同一 shell 生效）
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

# 显存监控函数
def print_vram(tag=""):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"[VRAM]{tag}: {mem:.2f} MB")

# 1) tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 加载模型（fp16 + 自动分权 + CPU/disk offload）
os.makedirs(OFFLOAD_DIR, exist_ok=True)
max_memory = {0: GPU_MEM_LIMIT, "cpu": CPU_MEM_LIMIT}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # RDNA2/ROCm 推荐 fp16
    #device_map="auto",           # GPU/CPU 自动分配， plan1-None
    max_memory=max_memory,       # 控制每个设备的内存上限
    offload_folder=OFFLOAD_DIR,  # 溢出到磁盘

    device_map=None,
    low_cpu_mem_usage=False #plan1修改F 原T

    #low_cpu_mem_usage=True



)

# 3) 应用 LoRA —— 两种方案二选一
if USE_PLAN_1:
    # === 方案 1: CPU 上加 LoRA 再搬回 GPU ===
    print(">> Using PLAN 1: Apply LoRA on CPU to avoid ROCm dtype bug...")
    model = model.cpu()
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    # 搬回 GPU (fp16)
    model = model.to(torch.float16).to("cuda")
else:
    # === 方案 2: 强制 LoRA dtype 为 fp16，避免 PEFT dtype 转换 ===
    print(">> Using PLAN 2: Force fp16 dtype during LoRA init...")
    from peft.tuners.tuners_utils import BaseTuner
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        dtype=torch.float16
    )
    # 强制 PEFT 在初始化适配器时使用 fp16，降低 dtype cast 风险
    BaseTuner._prepare_adapter_init_dtype = lambda self, model, dtype: torch.float16
    model = get_peft_model(model, lora_cfg)

# 4) 数据模板（Llama-3 chat 模板）
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
# def template(ex):
#     text = (f"<|start_header_id|>user<|end_header_id|>\n{ex['prompt']}<|eot_id|>"
#             f"<|start_header_id|>assistant<|end_header_id|>\n{ex['response']}<|eot_id|>")
#     return tokenizer(text, truncation=True, max_length=512)

def template(ex):
    text = (f"<|start_header_id|>user<|end_header_id|>\n{ex['prompt']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{ex['response']}<|eot_id|>")
    tokenized = tokenizer(text, truncation=True, max_length=512)
    # labels 直接等于 input_ids，这样 Trainer 会计算 causal LM loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


dataset = dataset.map(template, remove_columns=dataset.column_names)

# 5) 训练参数（adamw_torch 比 paged_* 更稳）
# args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     num_train_epochs=1,
#     learning_rate=2e-4,
#     fp16=True,                 # RDNA2/ROCm 下用 fp16
#     logging_steps=10,
#     save_steps=50,
#     optim="adamw_torch",
#     report_to=None
# )


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,              # ⚠️ 改为 False
    bf16=False,              # 确保没有 bf16
    logging_steps=10,
    save_steps=50,
    optim="adamw_torch",
    report_to=None
)






# 自定义 Trainer，加显存监控
# class VramTrainer(Trainer):
#     def training_step(self, model, inputs):
#         step = self.state.global_step
#         if step % LOG_VRAM_EVERY_STEPS == 0:
#             print_vram(f" step {step}")
#         return super().training_step(model, inputs)

class VramTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step
        if step % LOG_VRAM_EVERY_STEPS == 0:
            print_vram(f" step {step}")
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)





trainer = VramTrainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print("✅ LoRA 训练完成，权重保存在", OUTPUT_DIR)
