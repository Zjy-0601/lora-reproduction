# -*- coding: utf-8 -*-
"""
扩展实验 — Full Fine-tuning(全量微调)基线,仇恨言论检测任务

原论文只在 SST-2 上验证;本脚本把 Full FT 迁移到仇恨言论检测(tweet_eval/hate,二分类),
作为 train_lora_hate.py(LoRA)的对比基线,验证"LoRA 在新任务上性能接近 Full FT"。

与 train_full_ft.py 的区别:仅"读数据"这一段不同
  - train_full_ft.py      : 读本地 data/SST-2/*.parquet(列 sentence/label)
  - train_full_ft_hate.py : datasets 库加载 tweet_eval/hate(列 text/label)
其余(tokenizer / 训练循环 / 评估 / 参数量统计)完全一致。

跑法(在项目根目录,首次会联网下载 tweet_eval 数据集):
  ./.venv/Scripts/python.exe train/train_full_ft_hate.py
"""
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# ============ 1. 超参数(和 SST-2 主线保持一致) ============
MODEL_DIR = "models/roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5               # Full FT 用更小学习率(和 train_full_ft.py 一致)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ============ 2. 读数据(唯一改动点 ①:换数据集) ============
ds = load_dataset("cardiffnlp/tweet_eval", "hate")
train_df = ds["train"].to_pandas()
dev_df = ds["validation"].to_pandas()
print(f"train {len(train_df)} | dev {len(dev_df)}")

# ============ 3. tokenizer + 模型 ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.to(device)

# ============ 4. Dataset / DataLoader(唯一改动点 ②:列名 sentence → text) ============
class HateDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["text"].tolist(),
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


train_loader = DataLoader(HateDataset(train_df, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(HateDataset(dev_df, tokenizer, MAX_LEN),
                        batch_size=BATCH_SIZE)

# ============ 5. 统计参数量 ============
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / total: {total:,}")

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

# ============ 7. 训练循环(和 train_full_ft.py 完全一致) ============
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
print("\n========== Full FT on hate result ==========")
print(f"dev accuracy      : {final_acc:.4f}")
print(f"trainable params  : {trainable:,}")
print(f"peak GPU mem      : {peak_mem:.1f} MB")
print(f"train time        : {elapsed:.1f} s")
