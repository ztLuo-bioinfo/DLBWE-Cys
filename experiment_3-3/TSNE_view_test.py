import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 加载数据的函数
def load_npy(file_path):
    return np.load(file_path, allow_pickle=True)

# 整合特征数据
def consolidate_features(data):
    # 假设数据结构为 [folds][batches][samples, features...]
    # 我们首先将所有batch中的数据合并，然后将所有fold中的数据合并
    consolidated = np.concatenate([np.concatenate(fold, axis=0) for fold in data], axis=0)
    return consolidated

# 加载特征和标签数据
att = load_npy('./att_test.npy')
bilstm = load_npy('./BiLSTM_test.npy')
cnn = load_npy('./CNN_test.npy')
fnn = load_npy('./FNN_test.npy')
input_feature = load_npy('./input_feature_test.npy')
labels = load_npy('./label_test.npy')

# 整合并重塑特征数据，确保第一维为1212
att_consolidated = consolidate_features(att).reshape(913, -1)
bilstm_consolidated = consolidate_features(bilstm).reshape(913, -1)
cnn_consolidated = consolidate_features(cnn).reshape(913, -1)
fnn_consolidated = consolidate_features(fnn).reshape(913, -1)
input_feature_consolidated = consolidate_features(input_feature).reshape(913, -1)

# 打印每个整合后的特征集的形状以确认
print("Consolidated feature shapes:")
print(f"Attention: {type(att_consolidated)}")
print(f"BiLSTM: {type(bilstm_consolidated)}")
print(f"CNN: {type(cnn_consolidated)}")
print(f"FNN: {type(fnn_consolidated)}")
print(f"Input Feature: {type(input_feature_consolidated)}")
print(f"Labels: {labels.shape}")

# tsne_input_feature = TSNE(n_components=2, random_state=33)
# combined_input_feature_tsne = tsne_input_feature.fit_transform(input_feature_consolidated)

# tsne_cnn = TSNE(n_components=2, random_state=33)
# combined_cnn_tsne = tsne_cnn.fit_transform(cnn_consolidated)

# tsne_bilstm = TSNE(n_components=2, random_state=33)
# combined_bilstm_tsne = tsne_bilstm.fit_transform(bilstm_consolidated)
#
# tsne_att = TSNE(n_components=2, random_state=33)
# combined_att_tsne = tsne_att.fit_transform(att_consolidated)

tsne_fnn = TSNE(n_components=2, random_state=33)
combined_fnn_tsne = tsne_fnn.fit_transform(fnn_consolidated)

# colors = ['peachpuff', 'lightblue', 'plum', 'lightgreen']  # 浅色系列
# colors1 = ['orange', 'cyan', 'green', 'lightgreen']  # 浅色系列
# colors1 = ['orange', 'salmon', 'lightgreen', 'lightskyblue']

colors = {0: 'grey', 1: 'lightskyblue'}

# 创建一个简单的二元色条
cmap = mcolors.ListedColormap([colors[0], colors[1]])

# 可视化t-SNE结果的函数，现在包含简单的二元色条
def plot_tsne(tsne_result, labels, title):
    plt.figure(figsize=(8, 6))

    # 绘制散点图，使用标签数组来给点着色
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=cmap, alpha=0.6)

    # 创建色条
    colorbar = plt.colorbar(scatter, ticks=[0.25, 0.75])  # 设置色条上的刻度位置为四分位
    colorbar.set_ticklabels(['0', '1'])
    colorbar.ax.tick_params(labelsize=15)  # 设置色条标签的字体大小

    # 设置色条的颜色
    # colorbar.ax.yaxis.label.set_color('black')
    # colorbar.ax.yaxis.label.set_fontsize(12)

    plt.title(title, fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)

    plt.xlabel("")
    plt.ylabel("")

    # plt.savefig('./TSNE_feature_test.jpg', dpi=2000)
    # plt.savefig('./TSNE_feature_test.pdf', dpi=2000)
    # plt.savefig('./TSNE_cnn_test.jpg', dpi=2000)
    # plt.savefig('./TSNE_cnn_test.pdf', dpi=2000)
    # plt.savefig('./TSNE_bilstm_test.jpg', dpi=2000)
    # plt.savefig('./TSNE_bilstm_test.pdf', dpi=2000)
    # plt.savefig('./TSNE_att_test.jpg', dpi=2000)
    # plt.savefig('./TSNE_att_test.pdf', dpi=2000)

    plt.show()

# 对每个特征集进行可视化
# plot_tsne(combined_input_feature_tsne, labels, "Feature")
# plot_tsne(combined_cnn_tsne, labels, "CNN")
# plot_tsne(combined_bilstm_tsne, labels, "BiLSTM")
# plot_tsne(combined_att_tsne, labels, "Attention")
plot_tsne(combined_fnn_tsne, labels, "t-SNE Visualization of FNN Features")
