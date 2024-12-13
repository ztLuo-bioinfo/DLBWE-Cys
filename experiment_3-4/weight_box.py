import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 路径到.npy文件


data_path = './att_weight.npy'

# 加载数据并处理
att = np.load(data_path, allow_pickle=True)
processed_data = []

for d1 in att:
    for d2 in d1:
        for d3 in d2:
            processed_data.append(d3)

# 将数据重新整形为1212*5行，每行14列
att_list = np.array(processed_data).reshape(1204*5,-1)

# 绘制箱形图
plt.figure(figsize=(12, 4))
sns.boxplot(data=att_list, showfliers=False)  # 不显示离群值以获得更清晰的可视化效果
plt.title('Boxplot of the 14 Positions')
plt.xlabel('Position')
plt.ylabel('Values')
plt.xticks(ticks=np.arange(17), labels=np.arange(1, 18))  # 位置标签从1开始，不是0
plt.grid(True)
plt.savefig('./Box_weights.jpg',dpi=2000)
plt.savefig('./Box_weights.pdf',dpi=2000)
plt.show()

