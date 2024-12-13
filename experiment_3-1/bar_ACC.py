import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = pd.read_excel('./prediction_7model_ind.xlsx')
accuracy = (np.array(file.iloc[:,1])/100).tolist()
# 设置数据
models = ['DLBWE-Cys', 'CNN-BiLSTM', 'CNN', 'BiLSTM', 'SVM', 'RF', 'XGBoost']
# colors = ['peachpuff', 'lightblue', 'plum', 'lightgreen']  # 浅色系列


plt.figure(figsize=(10, 6))

ax = plt.gca()  # 获取当前的Axes对象
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框

# 创建一个 bar chart
# plt.bar(models, accuracy, color='salmon')
plt.bar(models, accuracy, color='peachpuff', width=0.5)  # 设置统一的颜色和宽度


plt.ylim(0.65,0.86)
# 设置标题和标签
plt.title('')
plt.xlabel('')
plt.ylabel('ACC',size=25)

plt.gcf().subplots_adjust(bottom=0.18)

plt.xticks(size=15, rotation=30, rotation_mode="anchor", ha="right")
plt.yticks(size=15)

# ax.spines['left'].set_linewidth(0.5)
# ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('./ind_ACC.jpg',dpi=2000)
plt.savefig('./ind_ACC.pdf',dpi=2000)
# 显示图表
plt.show()

