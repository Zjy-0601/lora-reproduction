# -*- coding: utf-8 -*-
"""
阶段 5 — LoRA(r=8)微调
在 SST-2 上用 LoRA 微调 RoBERTa-base,记录 4 项指标。

与 train_full_ft.py 的唯一区别:加载模型后注入 LoRA(只训练 Q/V 的低秩矩阵 A/B),
预训练的权重全部冻结。其余(读数据 / Dataset / 训练循环 / 评估)完全一致。

跑法(在项目根目录):
  ./.venv/Scripts/python.exe train/train_lora.py
"""
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd

# ============ 1. 超参数 ============
MODEL_DIR = "models/roberta-base"
DATA_DIR = "data/SST-2"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4                # LoRA 常用更大的学习率(因为只训练很小的 A/B 矩阵)
LORA_R = 8               # rank r
LORA_ALPHA = 8           # 缩放 α(论文:设成第一个 r,不调,即 α=r → α/r=1)
LORA_DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ============ 2. 读数据 ============
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
dev_df = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
print(f"train {len(train_df)} | dev {len(dev_df)}")

# ============ 3. tokenizer + 模型 + 注入 LoRA ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

# 关键区别 ①:注入 LoRA(其余代码和 Full FT 一模一样)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,               # 序列分类任务
    r=LORA_R,                                  # 低秩 r
    lora_alpha=LORA_ALPHA,                     # 缩放 α
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "value"],         # 只加在 Q 和 V(论文 5.2 的做法)
)
model = get_peft_model(model, lora_config)
model.to(device)

# ============ 4. Dataset / DataLoader(和 Full FT 完全相同) ============
class SST2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["sentence"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][i]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][i]),
            "label": torch.tensor(self.labels[i]),
        }


train_loader = DataLoader(SST2Dataset(train_df, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(SST2Dataset(dev_df, tokenizer, MAX_LEN),
                        batch_size=BATCH_SIZE)

# ============ 5. 统计参数量 ============
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
lora_only = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
print(f"trainable params: {trainable:,} (LoRA A/B: {lora_only:,}) / total: {total:,}")

# ============ 6. 评估函数 ============
def evaluate(model, loader):
    model.eval()
    correct = total_num = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["label"].to(device)
            correct += (preds == labels).sum().item()
            total_num += labels.size(0)
    return correct / total_num

# ============ 7. 训练循环 ============
# 关键区别 ②:optimizer 只收 requires_grad=True 的参数(冻结的 1.25 亿参数不进优化器,
# 这正是 LoRA 省显存/省存储的来源之一)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device),
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if step % 500 == 0:
            print(f"  epoch {epoch+1} step {step} loss {loss.item():.4f}")

    acc = evaluate(model, dev_loader)
    print(f"epoch {epoch+1} done | avg loss {epoch_loss/len(train_loader):.4f} | dev acc {acc:.4f}")

elapsed = time.time() - start
peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

# ============ 8. 输出 4 项指标 ============
final_acc = evaluate(model, dev_loader)
print("\n========== LoRA (r=8) result ==========")
print(f"dev accuracy      : {final_acc:.4f}")
print(f"trainable params  : {trainable:,}  (LoRA A/B: {lora_only:,})")
print(f"peak GPU mem      : {peak_mem:.1f} MB")
print(f"train time        : {elapsed:.1f} s")
