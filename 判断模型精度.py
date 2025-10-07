#判断模型是什么精度（fp16/fp32/bf16?
from safetensors.torch import load_file
from tkinter import filedialog
import tkinter as tk
root = tk.Tk()
root.withdraw()
file_ds = filedialog.askopenfilename(title="选择.safetensors，选好了点第一个")
print("你选择了：", file_ds)


tensors = load_file(file_ds)#("model-00001-of-00002.safetensors")
for name, tensor in tensors.items():
    print(name, tensor.dtype)
    break
