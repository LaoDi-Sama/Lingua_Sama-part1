# Q4_1：4-bit 中最推荐，精度稍高，聊天质量几乎无明显下降
#
# Q5_0 / Q5_K：如果你不在意显存，聊天质量更稳定



# Allowed quantization types:

#    2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B
#    3  or  Q4_1    :  4.78G, +0.4511 ppl @ Llama-3-8B
#------------------------------------------------------
#   38  or  MXFP4_MOE :  MXFP4 MoE
#    8  or  Q5_0    :  5.21G, +0.1316 ppl @ Llama-3-8B
#    9  or  Q5_1    :  5.65G, +0.1062 ppl @ Llama-3-8B
#   19  or  IQ2_XXS :  2.06 bpw quantization
#   20  or  IQ2_XS  :  2.31 bpw quantization
#   28  or  IQ2_S   :  2.5  bpw quantization
#   29  or  IQ2_M   :  2.7  bpw quantization
#   24  or  IQ1_S   :  1.56 bpw quantization
#   31  or  IQ1_M   :  1.75 bpw quantization
#   36  or  TQ1_0   :  1.69 bpw ternarization
#   37  or  TQ2_0   :  2.06 bpw ternarization
#   10  or  Q2_K    :  2.96G, +3.5199 ppl @ Llama-3-8B
#   21  or  Q2_K_S  :  2.96G, +3.1836 ppl @ Llama-3-8B
#   23  or  IQ3_XXS :  3.06 bpw quantization
#   26  or  IQ3_S   :  3.44 bpw quantization
#   27  or  IQ3_M   :  3.66 bpw quantization mix
#------------------------------------------------------
#   12  or  Q3_K    : alias for Q3_K_M
#   22  or  IQ3_XS  :  3.3 bpw quantization
#   11  or  Q3_K_S  :  3.41G, +1.6321 ppl @ Llama-3-8B
#   12  or  Q3_K_M  :  3.74G, +0.6569 ppl @ Llama-3-8B
#   13  or  Q3_K_L  :  4.03G, +0.5562 ppl @ Llama-3-8B
#   25  or  IQ4_NL  :  4.50 bpw non-linear quantization
#   30  or  IQ4_XS  :  4.25 bpw non-linear quantization
#   15  or  Q4_K    : alias for Q4_K_M
#   14  or  Q4_K_S  :  4.37G, +0.2689 ppl @ Llama-3-8B
#   15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B
#   17  or  Q5_K    : alias for Q5_K_M
#   16  or  Q5_K_S  :  5.21G, +0.1049 ppl @ Llama-3-8B
#   17  or  Q5_K_M  :  5.33G, +0.0569 ppl @ Llama-3-8B
#   18  or  Q6_K    :  6.14G, +0.0217 ppl @ Llama-3-8B
#    7  or  Q8_0    :  7.96G, +0.0026 ppl @ Llama-3-8B
#    1  or  F16     : 14.00G, +0.0020 ppl @ Mistral-7B
#   32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
#    0  or  F32     : 26.00G              @ 7B



from tkinter import filedialog
import tkinter as tk


root1 = tk.Tk()
root1.withdraw()

mode = 1
file_ds = filedialog.askopenfilename(title="选择要量化的gguf，选好了点第一个")
print(f"你选择了{file_ds}\n")

mds = {1: 'Q4_K', 2: 'Q4_0', 3: 'Q4_1',4:'Q5_0',5:'Q5_K'}
print(mds)
mode = int(input("选择量化"))

s = f'llama-quantize {file_ds} {file_ds}-{mds[mode]}.gguf {mds[mode]}'

'''llama-quantize \
  /home/laodi/LLM_Lora_Trainer/merged_model/Lingua20250920-212741-f16.gguf \
  /home/laodi/LLM_Lora_Trainer/quantized_output/Lingua-Q4_0.gguf \
  Q4_0'''


print(s)
#print('cd ~/llama.cpp/\n'+s+"\n")

print(f"文件将保存在：{file_ds}")

