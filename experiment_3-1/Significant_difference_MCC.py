import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
import numpy as np

class_list = ['DLEWB-Cys', 'CNN-BiLSTM', 'CNN', 'BiLSTM', 'XGBoost', 'RF', 'SVM']
classes1 = np.array(class_list)
classes = np.repeat(classes1,10)
# print(len(classes1))
index_list = []
for i in range(len(class_list)):
    file_name = './/{}_train_ave+-SD.xlsx'.format(class_list[i])
    file = pd.read_excel(file_name)
    accuracy = (np.array(file.iloc[:10, 4])).tolist()
    index_list += accuracy

index_array = np.array(index_list)

# data1 = {
#     'Class': np.random.choice(classes, 200),
#     'Value': np.random.uniform(0.74, 0.825, 200)
# }

data = {
    'Class': classes,
    'Value': index_array
}
print(data)
# print(data1)

mydata = pd.DataFrame(data)
mydata['Class'] = pd.Categorical(mydata['Class'], categories=class_list, ordered=True)

# 设置颜色
colors = ["#FC8D62", "#F2CC8E", "#82B29A", "#8DA0CB", "#99CC66", "#888888", "#008800", "#CC0033"]

# 绘制箱形图，增大图的尺寸
plt.figure(figsize=(10, 6))
sns.boxplot(showfliers=False, x='Class', y='Value', data=mydata, palette=colors, notch=False, linewidth=1.5)

# sns.swarmplot(x='Class', y='Value', data=mydata, color='black', alpha=0.75, size=10)

# 定义显著性水平
def significance_label(p):
    if p > 0.05:
        return 'NS'
    elif p < 0.05 and p >= 0.01:
        return '*'
    elif p < 0.01 and p >= 0.001:
        return '**'
    else:
        return '***'

# 计算并添加显著性标记
compaired = [("DLEWB-Cys", "CNN-BiLSTM"), ("DLEWB-Cys", "CNN"),
             ("DLEWB-Cys", "BiLSTM"), ("DLEWB-Cys", "XGBoost"), ("DLEWB-Cys", "RF"), ("DLEWB-Cys", "SVM")]

# 调整显著性标记的y位置和高度
max_value = mydata['Value'].max()
y = max_value + (max_value - mydata['Value'].min()) * 0.04  # 初始高度略高于最大值
h = (max_value - mydata['Value'].min()) * 0.1  # 控制显著性标记之间的垂直间隔
for i, pair in enumerate(compaired):
    # print(pair)
    # print(i)
    data1 = mydata[mydata['Class'] == pair[0]]['Value']
    # print(data1,'111')

    data2 = mydata[mydata['Class'] == pair[1]]['Value']
    # print(data2,'222')
    _, pvalue = mannwhitneyu(data1, data2)
    print(pvalue)
    sig_label = significance_label(pvalue)
    x1, x2 = class_list.index(pair[0]), class_list.index(pair[1])
    plt.plot([x1, x1, x2, x2], [y + i*h, y + h*(i+0.5), y + h*(i+0.5), y + i*h], lw=1.5, c='k')
    plt.text((x1+x2)*.5, y + h*(i+0.1), sig_label, ha='center', va='bottom', color='k', fontsize=18)

# 优化y轴的范围以适应显著性标记
plt.ylim(0.27, y + len(compaired) * h + 0.02)  # 留出足够的空间来展示显著性标记
plt.title('')
plt.xlabel('')
plt.ylabel('MCC',size=25)
plt.xticks(size=18, rotation=30, rotation_mode="anchor", ha="right")
plt.yticks(size=18)

# plt.xticks(rotation=45)
plt.tight_layout()

# 保存图像

plt.savefig('./ind_MCC_dif.jpg',dpi=2000)
plt.savefig('./ind_MCC_dif.pdf',dpi=2000)

plt.show()
