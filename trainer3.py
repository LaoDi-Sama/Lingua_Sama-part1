#Trainer3.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 模型与数据路径
model_path = "./Llama-3.2-3B-Instruct"
data_path  = "./TrainingDatas/Trainning_data.jsonl"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型（纯 FP16 + CPU Offload）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",              # 自动分配到 GPU + CPU
    offload_folder="./offload_dir", # CPU权重缓存位置（防止爆内存）
    low_cpu_mem_usage=True
)

# 3. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 加载数据
dataset = load_dataset("json", data_files=data_path)["train"]

def template(ex):
    text = f"<|start_header_id|>user<|end_header_id|>\n{ex['prompt']}<|eot_id|>" \
           f"<|start_header_id|>assistant<|end_header_id|>\n{ex['response']}<|eot_id|>"
    return tokenizer(text, truncation=True, max_length=512)

dataset = dataset.map(template, remove_columns=dataset.column_names)

# 5. 训练参数
args = TrainingArguments(
    output_dir="./lingua-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    optim="adamw_torch",
    report_to=None
)

# 6. 训练
Trainer(model=model, args=args, train_dataset=dataset).train()
