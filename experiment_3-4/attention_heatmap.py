import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import seaborn as sns
list = []
# scaler = preprocessing.MinMaxScaler()

str_all = './att_weight.npy'
# str_train = './pro_41_weights0000{}_train{}.npy'.format(num_w)
# str_test = './pro_41_weights0000{}_test{}.npy'.format(num_w)
att = np.load(str_all, allow_pickle=True)
# att = np.load(str_test, allow_pickle=True)
# att = np.load(str_train, allow_pickle=True)

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

# 使用Matplotlib来创建热图
plt.figure(figsize=(12, 4))
plt.imshow(attention_scores,cmap='Blues', aspect='auto')

# x_points = range(1,41)
# print(x_points)
plt.colorbar()
plt.xticks(ticks=np.arange(40), labels=np.arange(1, 41))
plt.yticks([])
# plt.xlabel('Sequence Position')
# plt.axis('off')
# plt.ylabel('Attention Score')
plt.title('Attention Scores Heatmap')
# plt.savefig('./Hot_map_weights.jpg',dpi=2000)
# plt.savefig('./Hot_map_weights.pdf',dpi=2000)

plt.show()





