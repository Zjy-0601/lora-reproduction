import matplotlib.pyplot as plt
#阶段五的两个方法，一个全量微调；一个Lora
methods=['Full-tunning','LoRA(r=8)']
acc=[0.9289,0.9369]

plt.bar(methods, acc) #plt.bar（x，y）俩参数
plt.xlabel('Methods')  #x轴标签
plt.ylabel('Accuracy') #y轴标签 
plt.title('Full-tunning vs LoRA') #标题 

#压缩范围，因为俩数字都是在这个范围，没必要从一开始算
plt.ylim(0.90,0.95)
plt.savefig('plot/fig1_acc.png',dpi=300) #保存图片


