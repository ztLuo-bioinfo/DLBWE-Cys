import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from Bio import SeqIO
import torch.nn.utils.rnn as rnn_utils
import random
import os
np.set_printoptions(suppress=True)
from Binary_Weight import One_hot

import argparse

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def load_features_ind(sample_name, a):
    AA_all = []
    AA_name = []
    aa = 0

    for my_a in SeqIO.parse(sample_name, 'fasta'):
        my_aa = my_a.seq
        # if len(my_aa) == 41:
        #     aa += 1
        if len(my_aa) >= 21:
            nu1_num1 = 0

            front_seq = my_aa[0:20]
            end_seq = my_aa[-21:-1]
            seq_add = front_seq[::-1] + my_aa + end_seq[::-1]
            for nu1 in range(len(seq_add[20:-21])):
                if seq_add[nu1 + 20] == 'C':
                    aa += 1

    print('aa:',aa)
    # aa = len(list(seq))
    i = 0
    p = np.zeros((aa, 41, 20))
    pp = p.copy()

    # pp3 = p.copy()

    for my_pp in SeqIO.parse(sample_name,'fasta'):

        if len(my_pp.seq) >= 21:
            add_supp = 0
            front_seq1 = my_pp[0:20]
            end_seq1 = my_pp[-21:-1]
            seq_add1 = front_seq1[::-1] + my_pp + end_seq1[::-1]

            for nu1 in range(len(seq_add1[20:-21])):
                if seq_add1[nu1 + 20] == 'C':
                    AA = seq_add1[nu1:nu1+41]
                    AA_all.append(int(nu1 + 1))
                    pp[i] = np.array(One_hot().main(AA, a))
                    # pp3[i] = AA_ONE_HOT(AA)
                    add_supp += 1
                    i += 1

            for supp in range(add_supp):
                AA_name.append(str(my_pp.description))
    print('name:',len(AA_name))
    print('i:',i)
    xx_all = pp

    return AA_name, AA_all, xx_all



class BahdanauAttention(nn.Module):

    def __init__(self, in_features, hidden_units, num_task):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)
        # self.fc = nn.Linear(16, 41)
    def forward(self, hidden_states, values):
        # print(hidden_states)
        # print(hidden_states.shape)
        # print(values)
        # print(values.shape)
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        # print(hidden_with_time_axis)
        # print(hidden_with_time_axis.shape)

        # print(self.W1(values))
        # print(self.W1(values).shape)

        # print(type(self.W2(hidden_with_time_axis)))
        # print(self.W2(hidden_with_time_axis).shape)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        # print('1111')
        # print(score)
        # print(score.shape)
        attention_weights = nn.Softmax(dim=1)(score)

        # print(attention_weights)
        # print(attention_weights.shape)
        values = torch.transpose(values, 1, 2)  # transpose to make it suitable for matrix multiplication
        # print('2222')
        # print(values)
        # print(values.shape)
        context_vector = torch.matmul(values, attention_weights)
        # attention_weights = self.fc(torch.transpose(attention_weights, 1, 2))
        # attention_weights = torch.transpose(attention_weights, 1, 2)

        # print('3333')
        # print(context_vector)
        # print(context_vector.shape)
        context_vector = torch.transpose(context_vector, 1, 2)
        # print('4444')
        # print(context_vector.shape)
        return context_vector, attention_weights


class newModel1(nn.Module):
    def __init__(self, out_channels1, out_channels2, kernel_size, input_size, hidden_size, att_in_features, att_hidden_units, in_feature):
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.kernel_size = kernel_size
        self.in_feature = in_feature

        self.att_in_features = att_in_features
        self.att_hidden_units = att_hidden_units

        super().__init__()

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.Attention = BahdanauAttention(in_features=self.att_in_features, hidden_units=self.att_hidden_units, num_task=1)

        # self.fc = nn.Linear(167, self.L_out_features)

        # self.lstm_fc = nn.Linear(2624, 1056)
        # nn.Flatten(),

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=self.out_channels1, kernel_size=self.kernel_size, stride=1, padding=0),  # 32*36
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv1d(in_channels=self.out_channels1, out_channels=self.out_channels2, kernel_size=self.kernel_size, stride=1, padding=0),  # 32*33
            nn.MaxPool2d(kernel_size=2),  # 32*17

        )

        self.block3 = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.Linear(self.in_feature, 128),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.in_feature, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)

        )

    def forward(self, x0):

        # print(x0.shape)
        x1 = x0.permute(0, 2, 1)
        # x1 = torch.unsqueeze(x0, 1)
        # print(x1.shape)
        x2 = self.block2(x1)
        # print(x2.shape)
        x3 = x2.permute(0, 2, 1)
        # print('1111', x3.shape)
        x4, (hn, hc) = self.lstm(x3, None)
        hn = hn.view(x4.size()[0], x4.size()[-1])
        # print('2222', hn.shape)
        context_vector, attention_weights = self.Attention(hn, x4)
        # print('3333', context_vector.shape)

        x5 = nn.Flatten()(context_vector)
        # print(x5.shape)


        # print(c0.shape)

        # c1 = self.fc(c0)
        # all1 = torch.cat([x5, c0], dim=1)

        # print(all1.shape)
        # print(all1.shape)
        # x3 = nn.functional.relu(x3)
        # print(c4.shape, "c4.shape")
        # all1 = torch.cat([c3, x3], dim=1)
        # print(all1.shape, "all1.shape")
        end = self.block3(x5)

        # print(end.shape, "end.shape")
        # print(c3.size())
        # end = self.block3(all1)
        return end, attention_weights

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    if (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN) == 0:
        MCC = 0
    else:
        MCC = (TP * TN - FP * FN) / pow((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN), 0.5)
    return SN, SP, ACC, MCC

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-decay", action="store", dest='decay_a', required=True,
                        help="a")

    parser.add_argument("-test_fasta", action="store", dest='test_fasta', required=True,
                        help="test fasta file")

    parser.add_argument("-out_dir", action="store", dest='out_dir', required=True,
                        help="output directory")


    args = parser.parse_args()
    decay_a = args.decay_a
    test_fa = args.test_fasta

    a, out_channels1, out_channels2, kernel_size, input_size, hidden_size, att_in_features, att_hidden_units, in_feature, lr1 = 0.02, 64, 128, 4, 64, 64, 128, 64, 128, 0.0005

    device = torch.device("cpu")
    seed_torch(seed=1949)
    AA_name, xulie, x_test = load_features_ind(test_fa, float(decay_a))
    test_data = torch.tensor(x_test).to(torch.float32).to(device)
    test_dataset = Data.TensorDataset(test_data)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    PATH1 = './DLBWE-Cys.pth'
    net1 = newModel1(out_channels1, out_channels2, kernel_size, input_size, hidden_size, att_in_features, att_hidden_units, in_feature).to(device)

    state1 = torch.load(PATH1,map_location= 'cpu')
    net1.load_state_dict(state1['model_state'])

    net1.eval()
    y_predict_1 = []
    y_test_class = []
    write_tp = []
    with torch.no_grad():
        for feature1_test in test_iter:
            # print(len(feature1_test))
            # print(feature1_test)
            feature1_test = feature1_test[0]
            pred_1, attention_weights_test = net1(feature1_test)

            x_p1 = pred_1.data.cpu().numpy().tolist()

            for p_list1 in x_p1:
                y_predict_1.append(p_list1)

        y_predict_1 = np.array(y_predict_1)

        y_test_class = np.array(y_test_class)

        probs_end1 = (y_predict_1[:, 1]).flatten().tolist()
        print(probs_end1)
        y_predict_class_1 = np.argmax(y_predict_1, axis=1)
        print(y_predict_class_1)
        prediction_labels_end = []
        for num in range(len(y_predict_class_1)):
            judge = y_predict_class_1[num]
            if judge > 0.5:
                prediction_labels_end = '+'
            else:
                prediction_labels_end = '-'
            write_tp.append([AA_name[num], xulie[num], prediction_labels_end, str(y_predict_1[num])])

    # write_tp.append([AA_name[num], xulie[num], prediction_labels_end, str(y_predict_1[num])])
    df = pd.DataFrame(write_tp, columns=['seq', 'position', 'pred', 'prob'])
    model_path = args.out_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df.to_excel('./' + model_path + '/result.xlsx',index=False)


