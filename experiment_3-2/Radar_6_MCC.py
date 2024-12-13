import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex

# 加载Excel文件
file_path = './contract_feature.xlsx'
df = pd.read_excel(file_path)


# colors = ['peachpuff', 'lightblue', 'plum', 'lightgreen']  # 浅色系列
# colors1 = ['orange', 'cyan', 'green', 'lightgreen']  # 浅色系列

# Let's find colors that are a bit darker than the provided list.


# Darkening factor (less than 1 to darken, greater than 1 to lighten)

# darkening_factor = 0.8 # ACC
# darkening_factor = 0.75  #SN
darkening_factor = 1 #MCC

# Function to darken a color
def darken_color(color, factor):
    # Convert the color to RGBA
    rgba = to_rgba(color)
    # Darken the RGB values and leave the alpha unchanged
    darkened_rgba = (rgba[0] * factor, rgba[1] * factor, rgba[2] * factor, rgba[3])
    # Convert the darkened RGBA color back to a hex color
    return to_hex(darkened_rgba)

# Apply the darken_color function to each color in the list
darker_colors = darken_color('green', darkening_factor)


# 设置雷达图的标签
labels = df['model_name'].tolist()

# 创建等分的角度数组，为每个标签一个角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 添加第一个角度在数组的末尾以闭合雷达图

# 从DataFrame中提取MCC值
mcc_values = df['MCC'].values.flatten().tolist()
mcc_values += mcc_values[:1]  # 添加第一个MCC值在数组的末尾以闭合雷达图

# 设置雷达图的值范围和起始角度
# radar_min, radar_max = 0.62, 0.781  # AAC
# radar_min, radar_max = 0.5, 0.781  # SN
radar_min, radar_max = 0.1, 0.55   # MCC

start_angle = 0  # 设置雷达图的起始角度为垂直方向

# 创建雷达图对象
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_ylim(radar_min, radar_max)  # 设置雷达图的最大和最小值
ax.set_theta_offset(np.radians(start_angle))  # 设置雷达图的起始角度

# 设置雷达图的环线值和标签
ring_increment = (radar_max - radar_min) / 4  # 环线之间的值
rings = np.arange(radar_min, radar_max + ring_increment, ring_increment)
ring_labels = [str(round(r, 2)) for r in rings]
ax.set_yticks(rings)
ax.set_yticklabels(ring_labels)

# 绘制MCC值的雷达图
ax.plot(angles, mcc_values, color=darker_colors, linewidth=1.2)  # 绘制线条
ax.fill(angles, mcc_values, color=darker_colors, alpha=0.3)  # 填充颜色

# 设置雷达图的标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)  # 设置较小的字号以避免重叠

# 微调标题位置以避免与最外环的值重叠
ax.set_title('MCC', size=15, color='k', position=(0.5, 1.15), horizontalalignment='center')

plt.savefig('./radar_train_MCC.jpg',dpi=2000)
plt.savefig('./radar_train_MCC.pdf',dpi=2000)

# 显示雷达图
plt.show()
