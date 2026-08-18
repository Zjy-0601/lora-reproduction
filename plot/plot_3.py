import matplotlib.pyplot as plt

ranks   = [1, 4, 8, 16]
seed42  = [0.9300, 0.9289, 0.9369, 0.9323]
seed123 = [0.9358, 0.9346, 0.9369, 0.9346]

fig, ax = plt.subplots(figsize=(7, 5))    # 只有一张小图,直接 fig, ax

ax.plot(ranks, seed42,  marker="o", label="seed 42")
ax.plot(ranks, seed123, marker="s", label="seed 123")

ax.set_xlabel("rank r")
ax.set_ylabel("dev accuracy")
ax.set_title("LoRA rank ablation (SST-2, qv)")
ax.set_ylim(0.925, 0.940)   # 收紧:0.929~0.937 的差异才看得见
ax.set_xticks(ranks)        # x 轴只标 1/4/8/16 这几个实测点
ax.legend()                 # 显示图例

fig.tight_layout()
fig.savefig("plot/fig2_rank.png", dpi=150)
