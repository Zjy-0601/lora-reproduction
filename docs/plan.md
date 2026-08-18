# LoRA 论文复现项目 — 执行计划

- 日期：2026-08-13
- 目标：2.5~3 周内完成一个可用于预推免/保研答辩的 LoRA 论文复现项目
- 最终路线：LoRA → RoBERTa-base → SST-2 → Full FT vs LoRA → rank 消融 → target modules 消融 → 实验分析 → GitHub + 报告 + 面试问答

## 0. 项目定位与交付物

**定位**：不是成为深度学习专家，而是项目完成后能真正理解代码、解释实验、分析结果，并在面试中独立讲清楚这个 LoRA 复现项目。

**三个交付物**：
1. GitHub 仓库（代码 + README + 实验对比图表）
2. 书面报告（复现 + 消融 + 分析）
3. 面试问答笔记（答辩演练产物）

## 1. 实验设计

- **模型**：RoBERTa-base（125M 参数，encoder 结构，分类任务自然）
- **数据集**：SST-2（GLUE 情感分类，二分类）。用 train 训练、dev 评估——SST-2 的 test 集无公开标签，论文报的 95.1 就是 dev 分数。
- **核心实验**：
  1. Full Fine-tuning（baseline）
  2. LoRA r=1 / r=4 / r=8 / r=16
- **第二优先级实验**：LoRA target modules 消融（只加注意力 q/v vs 全部相关线性层）
- **指标**：Accuracy + Trainable Parameters + GPU 显存占用 + 训练时间
- **论文 95.1 仅作参考，不作 KPI**，重点是验证：① 参数量是否显著下降；② 性能是否接近 Full FT；③ rank 影响；④ target modules 影响。
- **报告必须区分**：论文原始结果（如 RoBERTa-base+LoRA 的 dev 95.1）vs 我们自己的实验结果，并分析两者可能存在差异的原因（数据划分、随机种子、超参、训练细节、实现/版本差异、算力等）。
- **控制变量**：数据集/划分/随机种子/learning rate/batch size/epoch/评估方法全部固定，只在消融时改 rank 或 target modules。

## 2. 分阶段执行计划

> 每阶段包含：学什么 / 为什么 / 学到什么程度 / 学习资料 / 动手任务 / 过关标准 / 耗时。
> 原则：**够用优先**，不堆资料，不深挖。

### 阶段 0：环境搭建 + git（约 0.5 天）

- **学什么**：装 Python + PyTorch(CUDA 版) + HuggingFace 库；git 的 init/add/commit/push。
- **为什么**：后面一切的前提，第一道实际门槛。
- **学到什么程度**：能 `import torch` 且 `torch.cuda.is_available()` 返回 True；能把一个文件 push 到 GitHub。
- **资料**：PyTorch 官方安装页（按 CUDA 版本选）；git 官方 quickstart。
- **动手任务**：建虚拟环境，装好库，跑通 `torch.cuda.is_available()`；建 GitHub 空仓库并 push 一个 README。
- **过关标准**：`torch.cuda.is_available()` 为 True；GitHub 上能看到你的仓库。
- **耗时**：0.5 天。

### 阶段 1：Python 最小必要知识（灵活，已有基础可快速过）

- **学什么**：基本语法、函数、类、import、list/dict、for/if、文件读写、基本 OOP（只学项目用得到的）。
- **为什么**：能读懂、改动项目代码，不被语法卡住。
- **学到什么程度**：能看懂一段包含函数/类/循环的训练代码，能改几个超参数。
- **资料**：菜鸟教程 Python3 或廖雪峰教程，只过列出的章节。
- **动手任务**：写一个读文件、统计词频、用类封装的小脚本。
- **过关标准**：能不看教程写出"定义一个类 + 循环 + 读文件"的脚本。
- **耗时**：已有基础可快速过（约 0.5~1 天），不必真花 1.5 天"学 Python"；以"能读懂项目代码"为准，不看耗时。

### 阶段 2：PyTorch 最小必要知识（约 1.5~2 天）

- **学什么**：Tensor、shape/dtype/device、nn.Module、forward、loss、backward、optimizer、train/eval、Dataset/DataLoader、基本训练循环。
- **为什么**：项目代码的骨架，也是面试可能问"训练循环怎么写的"基础。
- **学到什么程度**：能手写一个最简单的 MNIST 分类训练循环并跑起来。
- **资料**：主看 B站刘二大人《PyTorch深度学习实践》（BV1Y7411d7Ys）第 1~9 讲（到 MNIST 为止，边看边敲；10~13 讲 CNN/RNN 本项目用不到，跳过）；PyTorch 官方 60-min blitz 作文字版补充。
- **动手任务**：跑通一个最简训练循环（MNIST 级即可）。
- **过关标准**：能口头讲清"forward → loss → backward → optimizer.step()"这一圈在干嘛。
- **耗时**：1.5~2 天。
- **⚠️ 注意**：MNIST 只是用来理解训练循环的工具，**不要在 MNIST 上钻研**（不调参、不刷分）。跑通并理解 forward → loss → backward → optimizer.step() 后，立即进入阶段 3 Transformer。

### 阶段 3：Transformer + RoBERTa 结构（约 1.5~2 天）

- **学什么**：Embedding、Self-Attention、Q/K/V、Attention、Feed Forward、Transformer Encoder、RoBERTa 基本结构。
- **为什么**：LoRA 是加在 Transformer 的线性层上的，不懂结构就讲不清 LoRA 加在哪。
- **学到什么程度**：能画图讲清"输入 token → embedding → 多层 encoder(attention + FFN) → 分类头"这条链路。
- **资料**：Jay Alammar《The Illustrated Transformer》《The Illustrated BERT》；3Blue1Brown 的 attention 可视化视频。
- **动手任务**：用 HuggingFace 加载 RoBERTa-base，打印模型结构，指出其中哪些是线性层。
- **过关标准**：能回答"Q/K/V 是什么、attention 在算什么、RoBERTa 和 BERT 的差别（掩码方式不同）"。
- **耗时**：1.5~2 天。

### 阶段 4：精读 LoRA 论文 + 吃透公式（约 1~1.5 天）

- **学什么**：W' = W + ΔW，ΔW = BA，实际缩放 α/r；W/A/B 各是什么；为什么冻结 W 只训练 A/B；rank r、α 是什么、区别；为什么低秩假设可能有效。
- **为什么**：整个项目的核心，答辩的命根子。
- **学到什么程度**：能不看笔记，白板写出公式并逐项解释每个符号和每步动机。
- **资料**：LoRA 论文原文（[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)）；官方仓库 [microsoft/LoRA](https://github.com/microsoft/LoRA)；HuggingFace [PEFT 文档](https://huggingface.co/docs/peft)。
- **动手任务**：把公式自己推导/默写一遍，写一份"每个符号是什么"的注释。
- **过关标准**：能解释 ① 为什么冻结 W；② A/B 的初始化（B=0，A=随机）为什么这样；③ α 和 r 分别管什么；④ 为什么说"低秩够用"是经验/理论假设而非严格普适证明。
- **耗时**：1~1.5 天。

### 阶段 5：跑通复现（Full FT + LoRA）（约 1~2 天）

- **学什么**：HuggingFace transformers + PEFT 的 LoraConfig；怎么对 RoBERTa 加分类头做 SST-2 分类。
- **为什么**：拿到第一组"能对比"的结果。
- **学到什么程度**：能独立跑通 Full FT 和 LoRA（r=8）两条线，并各自记录 accuracy / 可训练参数量 / 显存 / 时间。
- **资料**：HuggingFace transformers 的 text classification 示例；PEFT 的 LoRA 示例。
- **动手任务**：跑通两条线，把指标记进实验日志表。
- **过关标准**：Full FT 和 LoRA 都出 accuracy，LoRA 可训练参数量远小于 Full FT，且 accuracy 接近。
- **耗时**：1~2 天。

### 阶段 6：消融实验（rank + target modules）（约 2~3 天）

- **学什么**：怎么改 LoraConfig 的 r 和 target_modules；怎么做控制变量的对比。
- **为什么**：这是**你自己的工作**，答辩最值钱的部分。
- **学到什么程度**：能独立完成 r=1/4/8/16 和 target modules 两组消融，并解释每个趋势。
- **资料**：PEFT LoraConfig 参数文档。
- **动手任务**：固定其他超参，只改 r 跑 4 组；再固定 r 只改 target_modules 跑 2 组；每组记全 4 个指标。
- **过关标准**：能解释 ① r 增大意味着什么、结果怎么变、为什么；② 不同 target modules 为什么效果不同；③ 每组结果记在实验日志里。
- **⚠️ 注意**：rank 实验**不是为了证明"r 越大越好"或任何预设结论**。最终结果如实分析——如果 r=1/4/16 差别不明显、或小 r 反而更好，就如实写出并解释可能原因。**不为了符合预期去调整或挑选结果**，如实汇报才是科研该有的样子。
- **耗时**：2~3 天。

### 阶段 7：实验分析 + 报告 + GitHub + 答辩演练（约 2~3 天）

- **学什么**：怎么把结果整理成对比表和图表；README 与报告怎么写；答辩怎么讲。
- **为什么**：把动手的成果转成能答辩、能提交的东西。
- **学到什么程度**：有一份完整报告 + 结构清晰的 GitHub 仓库 + 一份问答笔记。
- **资料**：（无新资料，主要是整理 + 我陪你演练）
- **动手任务**：① 生成对比表/图；② 写报告和 README（报告中单列一节：论文原始结果 vs 我们实验结果及差异原因）；③ 我扮面试官逐条追问，你练到讲顺。
- **过关标准**：能按"问题 → LoRA 方法 → 实验设计 → Full FT 对比 → rank 消融 → target modules 消融 → 发现 → 结论"的故事线，不看稿讲完整项目。
- **耗时**：2~3 天。

## 3. 必懂清单（答辩核心，贯穿全程）

必须能口头讲清：
1. W' = W + ΔW 里 W 是什么，为什么它被冻结。
2. ΔW = BA，A/B 的形状、初始化（B=0，A 随机）及原因。
3. 实际使用 (α/r)·BA 的缩放，α 与 r 的区别。
4. rank r 越大意味着什么、对效果/参数量/过拟合的影响。
5. 为什么 LoRA 参数量大幅下降。
6. 为什么性能可能接近 Full FT（低秩假设 + 内蕴维度解释）。
7. 为什么"低秩就够用"不能说成有严格普适证明——这是核心的经验性/理论假设。
8. LoRA 相对 Adapter 的优势：无额外推理延迟，推理时可把 BA 合并进 W。
9. 定位要准确：PEFT 中最经典、影响力最大的基础方法之一（还有 QLoRA/AdaLoRA/DoRA/Adapter/Prefix Tuning）。

## 4. MVP 兜底方案

若时间不足，按此优先级收敛：
1. Full FT + LoRA（一个 rank）+ 参数量/Accuracy 对比 + 报告。
2. 再加 r=1/4/8/16。
3. 再加 target modules 消融。
4. 再加更完整分析。

宁可比预期少做实验，也不要烂尾。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| 第 1 周基础没跟上（最大风险） | 阶段 0~4 是瓶颈，严格执行；卡住立刻问，不停留在原地 |
| 环境装不上（CUDA 问题） | 阶段 0 就解决，必要时用 Colab 兜底跑小实验 |
| 钻进无底洞（CUDA/FlashAttention 等） | 严格"够用优先"，不深入清单之外的东西 |
| 最后赶报告 | 边做边记实验日志，报告是顺出来的不是赶出来的 |
| 时间不够 | 走 MVP 兜底，不做烂尾 |

## 6. 学习资料汇总（够用优先，不额外堆）

- Python：菜鸟教程 / 廖雪峰（只学阶段 1 列的章节）
- PyTorch：刘二大人《PyTorch深度学习实践》B站（https://www.bilibili.com/video/BV1Y7411d7Ys ，看 1~9 讲）+ 官方 60-min blitz（https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html）
- Transformer：Jay Alammar《The Illustrated Transformer》《The Illustrated BERT》；3Blue1Brown attention 视频
- LoRA 论文：https://arxiv.org/abs/2106.09685
- LoRA 官方仓库：https://github.com/microsoft/LoRA
- HuggingFace PEFT：https://huggingface.co/docs/peft
- HuggingFace transformers：https://huggingface.co/docs/transformers
