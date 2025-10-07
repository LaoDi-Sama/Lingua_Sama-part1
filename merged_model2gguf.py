from tkinter import filedialog
import tkinter as tk


root1 = tk.Tk()
root1.withdraw()

mode = 1
file_ds = filedialog.askdirectory(title="选择merged_model的目录，选好了点第一个")
print(f"你选择了{file_ds}\n复制以下\n")

mds = {1: 'f16', 2: 'bf16', 3: 'f32'}
# mode = int(input("选择精度"，mds))
s = f'python3 convert_hf_to_gguf.py {file_ds} --outfile {file_ds}-{mds[mode]}.gguf --outtype {mds[mode]}'


print('cd ~/llama.cpp/\n'+s+"\n")

print(f"文件将保存在：{file_ds}")


