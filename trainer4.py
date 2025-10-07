# train_offload_fp16.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch, os

MODEL_PATH = "./Llama-3.2-3B-Instruct"
DATA_PATH  = "./TrainingDatas/Trainning_data.jsonl"

# 1) ROCm 在 RDNA2 上用 fp16，别用 bf16
torch.set_default_dtype(torch.float32)

# 2) tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 3) 限制显存 + 开启磁盘 offload（很重要）
max_memory = {0: "5GiB", "cpu": "28GiB"}   # RX6600 建议 4.5~5.5GiB 之间
offload_dir = "./offload_dir"
os.makedirs(offload_dir, exist_ok=True)

# 4) 加载模型（FP16 + 自动分权 + 磁盘缓存）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,     # RDNA2 推荐 fp16
    device_map="auto",             # GPU/CPU 自动分配
    max_memory=max_memory,         # 控制每个设备的内存上限
    offload_folder=offload_dir,    # 溢出到磁盘
    low_cpu_mem_usage=True
)

# 5) LoRA
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 6) 数据模板（Llama3 chat）
ds = load_dataset("json", data_files=DATA_PATH)["train"]
def tpl(ex):
    text = (f"<|start_header_id|>user<|end_header_id|>\n{ex['prompt']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{ex['response']}<|eot_id|>")
    return tok(text, truncation=True, max_length=512)
ds = ds.map(tpl, remove_columns=ds.column_names)

# 7) 训练参数（用标准 AdamW，去掉 paged_* 优化器）
args = TrainingArguments(
    output_dir="./lingua-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,                 # 在 RDNA2/ROCm 用 fp16
    logging_steps=10,
    save_steps=50,
    optim="adamw_torch",
    report_to=None
)

Trainer(model=model, args=args, train_dataset=ds).train()
