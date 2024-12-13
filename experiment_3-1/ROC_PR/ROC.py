import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D,MaxPooling1D
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import auc,roc_curve,confusion_matrix,accuracy_score,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score

file = pd.read_excel('../prediction_7model_ind.xlsx')
# f1 = np.array(file.iloc[:,5]).tolist()
# print(f1)
list = ['DLBWE-Cys', 'CNN-BiLSTM', 'CNN', 'BiLSTM', 'SVM', 'RF', 'XGBoost']

for i in range(len(file['AUROC'])):


    str_fpr1 = './{}_fpr.npy'.format(list[i])
    str_tpr1 = './{}_tpr.npy'.format(list[i])

    # str_fpr1 = './{}_recall.npy'.format(list[i])
    # str_tpr1 = './{}_precision.npy'.format(list[i])

    print(str_fpr1)
    print(str_tpr1)

    fpr1 = np.load(str_fpr1)
    tpr1 = np.load(str_tpr1)

    str_lable = '{},AUROC = %.4f'.format(list[i])

    index = file['AUROC'][i]

    plt.plot(fpr1, tpr1, lw=2, label=str_lable % index)


# plt.figure(figsize=(10, 10))

# 画ROC曲线

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.tick_params(axis='x', labelsize=14)  # 设置X轴刻度的字体大小
plt.tick_params(axis='y', labelsize=14)  # 设置X轴刻度的字体大小
plt.title('ROC curve', fontsize=14)
plt.legend(loc="lower right")
plt.savefig('./ind_AUROC.pdf',dpi=2000)
plt.savefig('./ind_AUROC.jpg',dpi=2000)
plt.show()


# 画PR曲线
# plt.xlim([0.0, 1.0])
# plt.ylim([0.5, 1.0])
# plt.xlabel('Recall', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.title('PR curve', fontsize=14)
# plt.tick_params(axis='x', labelsize=14)  # 设置X轴刻度的字体大小
# plt.tick_params(axis='y', labelsize=14)  # 设置Y轴刻度的字体大小
# plt.legend(loc="lower left")
# plt.savefig('./ind_AUPR.pdf',dpi=1000)
# plt.savefig('./ind_AUPR.jpg',dpi=1000)

# plt.show()


# fpr8 = np.load(r"D:\Pycharm\PyCharm Community Edition 2020.1.4\holle world\m6Am\m6Am41\结果\m6Am\DLm6Am_20_recall.npy")
# tpr8 = np.load(r"D:\Pycharm\PyCharm Community Edition 2020.1.4\holle world\m6Am\m6Am41\结果\m6Am\DLm6Am_20_precision.npy")
# # fpr8 = np.load(r"D:\Pycharm\PyCharm Community Edition 2020.1.4\holle world\m6Am\m6Am41\结果\ind_m6Am_inter_attention_fpr.npy")
# # tpr8 = np.load(r"D:\Pycharm\PyCharm Community Edition 2020.1.4\holle world\m6Am\m6Am41\结果\ind_m6Am_inter_attention_tpr.npy")
#
#
# plt.figure(figsize=(10, 10))
# plt.plot(fpr8, tpr8, label='DLm6Am,AUPR= %.4f' % roc_auc8)
