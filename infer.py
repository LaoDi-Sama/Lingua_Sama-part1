import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# -----------------------------
# 环境：让 ROCm 报错更可读（0/1）
# -----------------------------
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "1")

# -----------------------------
# 选择 LoRA 目录（保留你的交互）
# -----------------------------
lora_dir = None
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    lora_dir = filedialog.askdirectory(title="选择 LoRA 所在文件夹（包含 adapter_model.safetensors）")
except Exception:
    pass

if not lora_dir:
    # 无法弹窗或用户取消，降级为命令行输入
    lora_dir = input("请输入 LoRA 目录路径（包含 adapter_model.safetensors）: ").strip()
print("LoRA 目录：", lora_dir)

# -----------------------------
# 配置
# -----------------------------
base_model_path = "./Llama-3.2-3B-Instruct"   # 你的基础模型本地路径
dtype_after_merge = torch.float16             # 可选：torch.bfloat16 / torch.float16 / torch.float32

# ROCm 下 torch.cuda 即 HIP
assert torch.cuda.is_available(), "未检测到 ROCm/HIP GPU（torch.cuda.is_available()=False）"
device = "cuda"

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 1) 在 CPU 上加载基础模型（fp32）
#    你的原注释说“fp32”，但代码用了 float16，这里修正为 float32
# -----------------------------
print("1) 在 CPU 上加载基础模型 (fp32)...")
base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,   # 明确 FP32，避免在 CPU 上触发半精度路径
    device_map=None,
    low_cpu_mem_usage=True
)

# -----------------------------
# 2) 在 CPU 上加载并应用 LoRA（fp32）
#    同样保持 fp32，避免中途半精度导致的 ROCm 内核问题
# -----------------------------
print("2) 在 CPU 上加载并应用 LoRA (fp32)...")
model = PeftModel.from_pretrained(
    base,
    lora_dir,
    torch_dtype=torch.float32,
    is_trainable=False
)

# -----------------------------
# 3) 合并 LoRA 并释放适配器权重
# -----------------------------
print("3) 合并 LoRA 并释放适配器权重...")
model = model.merge_and_unload()  # 得到标准 CausalLM

# -----------------------------
# 4) 一次性搬到 GPU 并设置 dtype
#    关键修正：避免先 to(device) 再 to(dtype) 的“二次转换”触发 HIP 内核
# -----------------------------
print(f"4) 一次性搬到 {device} 并转换到 {dtype_after_merge} ...")
model = model.to(device=device, dtype=dtype_after_merge)
model.eval()

# -----------------------------
# （可选）编译 - 先注释掉，等跑稳再开
# -----------------------------
# try:
#     model = torch.compile(model)
# except Exception:
#     pass

# -----------------------------
# 生成函数
# -----------------------------
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

# -----------------------------
# 交互
# -----------------------------
if __name__ == "__main__":
    print("\n输入 exit/quit 退出。\n")
    while True:
        s = input("Prompt> ").strip()
        if s.lower() in {"exit", "quit"}:
            break
        print("\nOutput:\n" + generate_text(s) + "\n")
