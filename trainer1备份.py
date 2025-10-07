# train.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer ,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

model_path = "./Llama-3.2-3B-Instruct"   # 模型位置
data_path  = "./TrainingDatas/Trainning_data.jsonl" #大概是jsondata的位置？

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)



model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,               # 便利贴数量，不够再继续加料，最高就16吧
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files=data_path)["train"]
def template(ex):
    text = f"<|start_header_id|>user<|end_header_id|>\n{ex['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{ex['response']}<|eot_id|>"
    return tokenizer(text, truncation=True, max_length=512)
dataset = dataset.map(template, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="./lingua-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    optim="paged_adamw_32bit"
)
Trainer(model=model, args=args, train_dataset=dataset).train()

