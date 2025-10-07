from tkinter import filedialog
import tkinter as tk
root1 = tk.Tk()
root1.withdraw()
from safetensors import safe_open
import torch
import numpy as np

file_ds = filedialog.askopenfilename(title="选择safetensors，选好了点第一个")
print(f"你选择了{file_ds}\n复制以下\n")

lora_path = file_ds#"your_lora_adapter.safetensors"

with safe_open(lora_path, framework="pt", device="cpu") as f:
    print("包含的 tensors:", f.keys())
    for k in f.keys():
        tensor = f.get_tensor(k)
        arr = tensor.numpy()
        if np.isnan(arr).any():
            print(f"⚠️ {k} 里有 NaN!")
        elif np.isinf(arr).any():
            print(f"⚠️ {k} 里有 Inf!")
        else:
            print(f"✅ {k} 正常")
