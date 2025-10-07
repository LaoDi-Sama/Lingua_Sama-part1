import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tkinter as tk
from tkinter import filedialog

root1 = tk.Tk()
root1.withdraw()

#file_ds = filedialog.askopenfilename()
file_ds = filedialog.askdirectory(title="选择lora所在文件夹，选好了点第一个")
print("你选择了：", file_ds)


# -------------------------------
# 配置
# -------------------------------
base_model_name = "./Llama-3.2-3B-Instruct"  # LLaMA 3.2 基础模型
lora_model_path = file_ds       # 微调的 LoRA 权重
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 128
temperature = 0.7
top_p = 0.9

# -------------------------------v
# 加载 Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# -------------------------------
# 加载基础模型
# -------------------------------
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,

    device_map="auto",       # ROCm 下可以用 auto 或手动 to(device)
    #torch_dtype=torch.float16,

    torch_dtype=torch.bfloat16,
)

# -------------------------------
# 加载 LoRA
# -------------------------------
print("Applying LoRA...")
model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.float16)
model.eval()

# -------------------------------
# 推理函数
# -------------------------------
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------
# 主程序示例
# -------------------------------
if __name__ == "__main__":
    while True:
        prompt = input("Prompt> ")
        if prompt.lower() in ["exit", "quit"]:
            break
        output_text = generate_text(prompt)
        print(f"\nOutput:\n{output_text}\n")
