import matplotlib.pyplot as plt
import numpy as np

configs = ["qv", "qkv"]
seed42  = [0.9369, 0.9346]
seed123 = [0.9369, 0.9358]

x = np.arange(2)          # [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(x - width/2, seed42,  width, label="seed 42")
ax.bar(x + width/2, seed123, width, label="seed 123")

ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("dev accuracy")
ax.set_title("LoRA target_modules ablation (SST-2, r=8)")
ax.set_ylim(0.930, 0.940)   # 同样收紧:这两组差距只有 0.0017
ax.legend()

fig.tight_layout()
fig.savefig("plot/fig3_target.png", dpi=150)
