import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

# 关键：告诉 Tk 用哪个中文字体


file = filedialog.askopenfilename(title="选择文件，选好了点第一个")
print("你选择了：", file)