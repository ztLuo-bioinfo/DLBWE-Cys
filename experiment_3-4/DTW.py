
# from sklearn import svm
import numpy as np
from tslearn.metrics import dtw, dtw_path
from tslearn.metrics import soft_dtw
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
# from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import math

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
# str_train = './pro_41_weights0000{}_train{}.npy'.format(num_w)
# str_test = './pro_41_weights0000{}_test{}.npy'.format(num_w)
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
print(len(list))
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
vec_40 = attention_scores.reshape(40)


fig = plt.figure()
ax0=fig.add_subplot(1,1,1)

ax0.plot(ts,y1,'r',label='Attention weights')
ax0.plot(ts2,vec_40,'g',label='position weights')

optimal_path, dtw_score = dtw_path(y1, vec_40)
print(optimal_path)
plt.figlegend()
for match in optimal_path:
    ax0.plot([ts[match[0]],ts2[match[1]]],[y1[match[0]],vec_40[match[1]]])

plt.savefig('./DTW_weights_similarity.jpg',dpi=2000)
plt.savefig('./DTW_weights_similarity.pdf',dpi=2000)

plt.show()


