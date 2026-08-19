# -*- coding: utf-8 -*-
"""
扩展实验 — LoRA 在内容安全任务上的验证

原论文只在 GLUE 情感分类(SST-2)上验证 LoRA;本脚本把**同一套 LoRA 代码几乎不改**
迁移到仇恨言论检测(tweet_eval/hate,二分类),验证 LoRA 在"原论文没覆盖的安全向
文本任务"上同样有效 —— 这是本项目相对纯复现的一点扩展(不是科学创新,是"方法迁移到新场景")。

与 train_lora.py 的区别:仅"读数据"这一段不同
  - train_lora.py      : 读本地 data/SST-2/*.parquet(列 sentence/label)
  - train_lora_hate.py : datasets 库加载 tweet_eval/hate(列 text/label)
其余(tokenizer / 注入 LoRA / 训练循环 / 评估 / 参数量统计)完全一致。

跑法(在项目根目录,首次会联网下载 tweet_eval 数据集):
  ./.venv/Scripts/python.exe train/train_lora_hate.py                    # r=8(默认)
  ./.venv/Scripts/python.exe train/train_lora_hate.py --r 16 --seed 42   # r 消融
"""
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import argparse
import random
import numpy as np

# ============ 1. 超参数(和 SST-2 主线保持一致,便于对比) ============
MODEL_DIR = "models/roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4                # LoRA 常用更大的学习率
LORA_DROPOUT = 0.1

# 命令行参数(和 train_lora_ablation.py 对齐,便于在 hate 上做 r 消融)
parser = argparse.ArgumentParser()
parser.add_argument("--r", type=int, default=8, help="LoRA rank r")
parser.add_argument("--seed", type=int, default=42, help="random seed(控制变量)")
args = parser.parse_args()

LORA_R = args.r
LORA_ALPHA = LORA_R      # α 跟着 r 走,保证 α/r=1(只变 rank、不变缩放)

# 固定随机种子(控制变量,保证不同 r 可比)
SEED = args.seed
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ============ 2. 读数据(唯一改动点 ①:换数据集) ============
# 仇恨言论检测:二分类(0=非仇恨, 1=仇恨),列 text / label
ds = load_dataset("cardiffnlp/tweet_eval", "hate")
train_df = ds["train"].to_pandas()
dev_df = ds["validation"].to_pandas()
print(f"train {len(train_df)} | dev {len(dev_df)}")

# ============ 3. tokenizer + 模型 + 注入 LoRA(与 train_lora.py 完全一致) ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "value"],
)
model = get_peft_model(model, lora_config)
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

# ============ 7. 训练循环(与 train_lora.py 完全一致) ============
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
print("\n========== LoRA (r=8) on hate result ==========")
print(f"dev accuracy      : {final_acc:.4f}")
print(f"trainable params  : {trainable:,}  (LoRA A/B: {lora_only:,})")
print(f"peak GPU mem      : {peak_mem:.1f} MB")
print(f"train time        : {elapsed:.1f} s")
