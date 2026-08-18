import matplotlib.pyplot as plt

methods=['Full-tunning','LoRA(r=8)']
acc=[0.9289,0.9369]
params=[124647170,887042]
gpu_mem=[3164,1529]
time     = [2480, 1551]

#2行2列的子图 10x8英寸
fig,axes=plt.subplots(2,2,figsize=(10,8))

#左上 acc 收紧到0.90-0.95
axes[0,0].bar(methods, acc) #plt.bar（x，y）俩参数
axes[0,0].set_title('dev accuracy') #标题
axes[0,0].set_ylim(0.90,0.95) 
#压缩范围，因为俩数字都是在这个范围，没必要从一开始算


#右上可训练参数数量
axes[0,1].bar(methods, params) #plt.bar（x，y）俩参数
axes[0,1].set_title('Trainable Parameters') #标题
axes[0,1].set_yscale('log') #y轴对数刻度

#左下显存峰值
axes[1,0].bar(methods, gpu_mem) #plt.bar（x，y）俩参数
axes[1,0].set_title('GPU Memory(MB)') #标题
axes[1,0].set_ylim(0,3500) #y轴范围



#右下训练时间
axes[1,1].bar(methods, time) #plt.bar（x，y）俩参数
axes[1,1].set_title('Training Time(h)') #标题
axes[1,1].set_ylim(0,3000) #y轴范围


fig.suptitle("Full FT vs LoRA on SST-2")   # 整张大图的标题
fig.tight_layout()                          # 自动拉开子图间距,防止标签互相压
fig.savefig("plot/fig1_4dim.png", dpi=150)