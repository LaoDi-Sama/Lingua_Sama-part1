import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import tkinter as tk
from tkinter import filedialog

# 可选：让 ROCm 报错更可读
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "1")

# ---------- 选择 LoRA 目录 ----------
root = tk.Tk(); root.withdraw()
lora_dir = filedialog.askdirectory(title="选择 LoRA 所在文件夹（包含 adapter_model.safetensors）")
print("LoRA 目录：", lora_dir)

# ---------- 配置 ----------
base_model_path = "./Llama-3.2-3B-Instruct"   # 你的基础模型本地路径
dtype_after_merge = torch.float16            # 合并后搬到GPU用的精度：bfloat16 / float16 / float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 加载 tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

print("1) 在 CPU 上加载基础模型 (fp32)...")
base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,   # 先用 CPU 的 fp32，避免 ROCm 内核问题
    device_map=None
)

print("2) 在 CPU 上加载并应用 LoRA (fp32)...")
model = PeftModel.from_pretrained(
    base,
    lora_dir,
    torch_dtype=torch.float16,   # 同样保持 fp32
    is_trainable=False
)

print("3) 合并 LoRA 并释放适配器权重...")
model = model.merge_and_unload()              # 得到一个标准的 CausalLM 模型

print(f"4) 搬到 {device} 并转换到 {dtype_after_merge} ...")
model.to(device)
if dtype_after_merge != torch.float32:
    model = model.to(dtype_after_merge)

model.eval()

# （可选）编译，ROCm 6.3 有时能加速
try:
    model = torch.compile(model)
except Exception:
    pass

# ---------- 生成函数 ----------
gen_cfg = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ---------- 交互 ----------
if __name__ == "__main__":
    while True:
        s = input("Prompt> ")
        if s.strip().lower() in {"exit","quit"}:
            break
        print("\nOutput:\n" + generate_text(s) + "\n")
