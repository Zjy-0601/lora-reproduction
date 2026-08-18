# -*- coding: utf-8 -*-
"""
阶段 5 — Full Fine-tuning(全量微调)基线
在 SST-2 上全量微调 RoBERTa-base,记录 4 项指标:
  accuracy / 可训练参数量 / 显存峰值 / 训练时间

跑法(在项目根目录):
  ./.venv/Scripts/python.exe train/train_full_ft.py
"""
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# ============ 1. 超参数 ============
MODEL_DIR = "models/roberta-base"   # 本地 RoBERTa-base 权重
DATA_DIR = "data/SST-2"             # 本地 SST-2 数据
MAX_LEN = 128                       # 每条句子截断/补齐到 128 个 token
BATCH_SIZE = 16                     # 一个 batch 的样本数
EPOCHS = 3                          # 训练轮数
LR = 2e-5                           # 学习率(RoBERTa 微调常用 1e-5 ~ 3e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ============ 2. 读数据 ============
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
dev_df = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
print(f"train {len(train_df)} | dev {len(dev_df)}")

# ============ 3. tokenizer + 模型 ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.to(device)

# ============ 4. Dataset / DataLoader ============
class SST2Dataset(Dataset):
    """把 DataFrame 变成 PyTorch 能迭代的 Dataset。

    关键:tokenize 在 __init__ 里一次性批量做掉(快),
    __getitem__ 只负责按索引取第 i 条,不再重复 tokenize。
    """
    def __init__(self, df, tokenizer, max_len):
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["sentence"].tolist(),
            truncation=True,          # 超长截断
            padding="max_length",     # 不足则用 pad token 补齐到 max_len
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
            preds = outputs.logits.argmax(dim=-1)   # 取 logits 最大的那一类  二分类问题，预测取一个
            labels = batch["label"].to(device)
            correct += (preds == labels).sum().item()
            total_num += labels.size(0)
    return correct / total_num

# ============ 7. 训练循环(和 MNIST 同一个骨架) ============
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()   # 清零显存峰值,好统计本次训练的峰值
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
print("\n========== Full FT result ==========")
print(f"dev accuracy      : {final_acc:.4f}")
print(f"trainable params  : {trainable:,}")
print(f"peak GPU mem      : {peak_mem:.1f} MB")
print(f"train time        : {elapsed:.1f} s")
