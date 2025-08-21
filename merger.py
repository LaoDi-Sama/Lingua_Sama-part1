from transformers import LlamaForCausalLM
from peft import PeftModel
import torch
import tkinter as tk
from tkinter import filedialog
import time
current_time = time.strftime("%Y%m%d-%H%M%S")

root1 = tk.Tk()
root1.withdraw()

file_ds = filedialog.askdirectory(title="选择lora所在文件夹，选好了点第一个")
print("你选择了：", file_ds)


# 基础模型路径
base_model_path = "/home/laodi/LLM_Lora_Trainer/Llama-3.2-3B-Instruct"#########################model

# LoRA 权重路径
lora_model_path = file_ds

# 加载基础模型
model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map="cpu",
    torch_dtype=torch.float16,########################量化
    low_cpu_mem_usage=True
)

# 加载 LoRA 并合并
model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.float16)
model = model.merge_and_unload()

# 保存合并后的完整模型
output_path = "/home/laodi/LLM_Lora_Trainer/merged_model/Lingua%s" % current_time
model.save_pretrained(output_path, safe_serialization=True)




#config.json、tokenizer.json、tokenizer.model、special_tokens_map.json、generation_config.json（