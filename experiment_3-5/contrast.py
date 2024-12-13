import pandas as pd
import matplotlib.pyplot as plt

# 加载Excel文件
file_path_no_weight = './21-61_contrast_no_weight.xlsx'
file_path_weight = './21-61_contrast_weight.xlsx'

# 读取数据
data_no_weight = pd.read_excel(file_path_no_weight)
data_weight = pd.read_excel(file_path_weight)

# 定义x轴的固定点
x_points = [21, 31, 41, 51, 61]

# 获取每个文件对应的ACC值，并转换为百分比
y_no_weight = data_no_weight['ACC'].values / 100
y_weight = data_weight['ACC'].values / 100

# 绘制折线图
plt.figure(figsize=(6, 3))
plt.plot(x_points, y_weight, label='Weight', marker='o')  # 有权重数据
plt.plot(x_points, y_no_weight, label='No Weight', marker='o')  # 无权重数据

# 设置x轴的刻度值
plt.xticks(x_points)

# 添加图例
plt.legend()

# 设置图表标题和坐标轴标签
# plt.title('ACC Values by Model')
# plt.xlabel('seqence length')
plt.ylabel('ACC')

# 显示图表
plt.tight_layout()
plt.savefig('./weights contract no weights.jpg',dpi=5000)
plt.savefig('./weights contract no weights.pdf',dpi=2000)
plt.show()
