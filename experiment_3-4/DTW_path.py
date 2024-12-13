import numpy as np
import matplotlib.pyplot as plt
import math

# 示例时间序列
# time_series_a = np.array([0, 1, 1, 2, 3, 2, 1])
# time_series_b = np.array([1, 1, 2, 2, 2, 3, 2, 1])

ts=np.linspace(0,40,41)
ts2=np.linspace(0,39,40)


stride = 0.2
a = 0.1
P0 = 1
y1 = []
for m in range(41):
    distence = float(abs(m - ((41 - 1) / 2)) * stride)
    weight = float(P0 * math.exp(-a * distence))
    y1.append(weight)
print(y1)
time_series_a = np.array(y1)

str_all = './att_weight.npy'
att = np.load(str_all, allow_pickle=True)
# att = np.load(str_test, allow_pickle=True)
# att = np.load(str_train, allow_pickle=True)
list = []
for d1 in att:
    for d2 in d1:
        for d3 in d2:
            # d3 = scaler.fit_transform(np.array(d3)).tolist()
            # print(d3)
            list.append(d3)
# att_list = np.array(list)

# print(att_list.shape)

att_list = np.array(list).reshape(1204*5,-1)
# att_list = np.array(list).reshape(1212,-1)
# att_list = np.array(list).reshape(1212*4,-1)

# print(fold1.shape)

# attention_scores = scaler.fit_transform(att_list)
# attention_scores = np.mean(attention_scores,axis=0).reshape(1, -1)

attention_scores = np.mean(att_list,axis=0).reshape(1,-1)
# print(attention_scores)
# attention_scores = scaler.fit_transform(attention_scores).reshape(1, -1)

# attention_scores = scaler.fit_transform(attention_scores)
print(attention_scores.shape)
print(attention_scores)

y2 = attention_scores[0]
y2 = np.repeat(y2,2)
print(y2)
vec_37 = np.zeros((37,1))
for i in range(34):
    vec_37[i:i+4] += y2[i] / 4

print(vec_37, '1111111')

# # 从37维到41维的逆过程
vec_40 = np.zeros((40, 1))
for i in range(37):
    vec_40[i:i+4] += vec_37[i] / 4
attention_scores = vec_40.reshape(1,-1)
print(attention_scores.shape)
time_series_b = attention_scores.reshape(40)


# 初始化DTW网格
dtw_grid = np.zeros((len(time_series_a), len(time_series_b)))

# 使用欧几里得距离填充DTW网格
for i in range(len(time_series_a)):
    for j in range(len(time_series_b)):
        cost = abs(time_series_a[i] - time_series_b[j])
        dtw_grid[i, j] = cost

# 计算累积距离
for i in range(1, len(time_series_a)):
    dtw_grid[i, 0] += dtw_grid[i - 1, 0]
for j in range(1, len(time_series_b)):
    dtw_grid[0, j] += dtw_grid[0, j - 1]
for i in range(1, len(time_series_a)):
    for j in range(1, len(time_series_b)):
        dtw_grid[i, j] += min(dtw_grid[i - 1, j], dtw_grid[i, j - 1], dtw_grid[i - 1, j - 1])

# 回溯找到路径
path = []
i = len(time_series_a) - 1
j = len(time_series_b) - 1
path.append((i, j))
while i > 0 and j > 0:
    min_index = np.argmin([dtw_grid[i - 1, j], dtw_grid[i, j - 1], dtw_grid[i - 1, j - 1]])
    if min_index == 0:
        i -= 1
    elif min_index == 1:
        j -= 1
    else:
        i -= 1
        j -= 1
    path.append((i, j))

# 反转路径
path = path[::-1]


checkerboard = (np.indices((40,41)).sum(axis=0) % 2)
# print(dtw_grid.shape)
# 可视化DTW矩阵
fig, ax = plt.subplots()
cmap = plt.get_cmap('gray')  # 创建一个灰度色图
ax.imshow(checkerboard, cmap=cmap, origin='lower', aspect='auto', alpha=0.3)  # alpha设置为半透明，以便查看背后的数据

plt.plot([p[0] for p in path], [p[1] for p in path], 'r', linewidth=3)
# plt.colorbar()
plt.xticks(range(0, len(time_series_a), 5), range(1, len(time_series_a)+1, 5))
plt.yticks(range(0, len(time_series_b), 5), range(1, len(time_series_b) + 1, 5))

# plt.xlabel('Time Series A')
# plt.ylabel('Time Series B')
# plt.title('DTW of two time series')
plt.savefig('./DTW_path_weights.jpg',dpi=2000)
plt.savefig('./DTW_path_weights.pdf',dpi=2000)

plt.show()


