from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

import csv
def aaindex(sequences):
# 打开CSV文件
  with open('aaindex_feature.csv', 'r') as file:
      csv_reader = csv.reader(file)

      header = next(csv_reader)


      data_dict = {}
      for row in csv_reader:
          name = row[0]  # 第一列是名称
          values = [float(value) for value in row[1:]]  # 后面的列为数值，转换为浮点数

          data_dict[name] = values

  code=[]
  for j in sequences:
      # print(j)
      code = code + data_dict[j]

  return code

def BLOSUM62(sequences):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '*': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # *
        '_': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # _
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # _
    }

    code = []
    for j in sequences:
        code = code + blosum62[j]
    return code

import os
import re

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
from transformers import BertModel, BertTokenizer
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'



def genData_typeA(file):
    df=pd.read_csv(file,encoding='utf-8-sig',header=None,skiprows=1,names=['seq','label'])
    pep_codes = []
    labels = []
    pep_seq = []
    seq=df['seq']
    label=df['label']
    for i in range(len(seq)):
      labels.append(int(label[i]))
      pep=seq[i]
      input_seq=' '.join(pep)
      input_seq = re.sub(r"[UZOB]", "X", input_seq)
      pep_seq.append(input_seq)
      current_pep=[]
      # print(pep)
      # for aa in pep:
      #   current_pep.append(aa_dict[aa])
      target_length=50
      while len(pep)<target_length:
          pep=pep+'X'
      # print(pep)
      aa_pep=aaindex(pep)
      aa_pep=torch.tensor(aa_pep).reshape(-1,531)
      blo_pep=BLOSUM62(pep)
      blo_pep=torch.tensor(blo_pep).reshape(-1,20)
      current_pep=torch.cat((aa_pep,blo_pep),dim=-1)
      print(current_pep.size())
      pep_codes.append(torch.tensor(current_pep))

def genData_typeB(file):
    df=pd.read_csv(file,encoding='utf-8-sig',header=None,skiprows=1,names=['seq','label'])
    pep_codes = []
    labels = []
    pep_seq = []
    seq=df['seq']
    label=df['label']
    for i in range(len(seq)):
      labels.append(int(label[i]))
      pep=seq[i]
      input_seq=pep
      pep=pep.replace(" ","")
      input_seq = re.sub(r"[UZOB]", "X", input_seq)
      pep_seq.append(input_seq)
      current_pep=[]
      # print(pep)
      # for aa in pep:
      #   current_pep.append(aa_dict[aa])
      target_length=50
      while len(pep)<target_length:
          pep=pep+'X'
      # print(pep)
      aa_pep=aaindex(pep)
      aa_pep=torch.tensor(aa_pep).reshape(-1,531)
      blo_pep=BLOSUM62(pep)
      blo_pep=torch.tensor(blo_pep).reshape(-1,20)
      current_pep=torch.cat((aa_pep,blo_pep),dim=-1)
      blo_pep=None
      aa_pep=None
      print(current_pep.size())
      pep_codes.append(torch.tensor(current_pep))


    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    print(data.size())

    return data, torch.tensor(labels), pep_seq



class MyDataSet(Data.Dataset):
    def __init__(self, data, label, bert_data):
        self.data = data
        self.label = label
        self.bert_data = bert_data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.bert_data[idx]

if __name__ == '__main__':
    train_data, train_label, train_seq = genData_typeB("train.csv")
    test_data, test_label, test_seq = genData_typeB("test.csv")
    print(train_seq)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to('cuda')

#Unsupervised learning
    seq1=train_seq
    seq2=test_seq

    ids = tokenizer.batch_encode_plus(seq1, add_special_tokens=True,max_length=50,pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to('cuda')
    attention_mask = torch.tensor(ids['attention_mask']).to('cuda')
    with torch.no_grad():
         out_train= bert(input_ids=input_ids,attention_mask=attention_mask)[0]



    ids = tokenizer.batch_encode_plus(seq2, add_special_tokens=True, max_length=50,pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to('cuda')
    attention_mask = torch.tensor(ids['attention_mask']).to('cuda')
    with torch.no_grad():
        out_test = bert(input_ids=input_ids,attention_mask=attention_mask)[0]

    torch.save(train_data,'feature_train.pth')
    torch.save(test_data,'feature_test.pth')
    torch.save(train_label,'label_train.pth')
    torch.save(test_label,'label_test.pth')
    torch.save(out_test,'BERT_test.pth')
    torch.save(out_train,'BERT_train.pth')
    torch.cuda.empty_cache()

class newModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 25
        self.emb_dim = 20

        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)
        self.bilstm = nn.LSTM(input_size=self.emb_dim,hidden_size=self.hidden_dim,num_layers=4,bidirectional=True,dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 32),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32, 1),
                                    )

        self.block1 = nn.Sequential(nn.Linear(2500, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048, 1024),
                                    )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x, pep):
        x=x.float()
        output=x.permute(1, 0, 2)
        #GRU
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        return self.block1(output)

    def trainModel(self, x, pep):
        with torch.no_grad():
            output = self.forward(x, pep)
            output = torch.cat((output,pep.reshape(output.size(0),-1)),dim=1)


        return self.block2(output)


class MultiContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(MultiContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label1, label2):
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(output1, output2.t())

        # 使用label信息创建mask矩阵
        # mask1 = torch.eye(output1.size(0)).bool().to(device)
        # mask2 = torch.eye(output2.size(0)).bool().to(device)
        pos_mask = (label1.unsqueeze(1) == label2.unsqueeze(0)).bool()
        neg_mask = ~(pos_mask)
        # print(label1,label2,label1.unsqueeze(1) ,label2.unsqueeze(0))

        # 计算正样本对的损失
        pos_pairs = similarity_matrix[pos_mask].view(-1, 1)
        pos_loss = torch.sum(torch.pow(pos_pairs - self.margin, 2))

        # 计算负样本对的损失
        # print(similarity_matrix[neg_mask].size())
        neg_pairs = similarity_matrix[neg_mask].view(-1, 1)
        neg_loss = torch.sum(torch.clamp(self.margin - neg_pairs, min=0.0))

        # 计算总损失
        loss = (pos_loss + neg_loss) / (pos_pairs.size(0) + neg_pairs.size(0))

        return loss

def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    pep1_ls = []
    pep2_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, pep_seq1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, pep_seq2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1], batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        pep1_ls.append(pep_seq1)
        pep2_ls.append(pep_seq2)
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2, pep1_ls, pep2_ls

def evaluate_accuracy(data_iter, net):
    prelabel, relabel = [], []
    acc_sum=0.0
    x_list=[]
    y_list=[]
    z_list=[]
    for x,y,z in data_iter:
      x_list.append(x.to('cuda'))
      y_list.append(y.to('cuda'))
      z_list.append(z.to('cuda'))
    # vec=data_iter[2]
    # x=data_iter[0]
    # y=data_iter[1]
    x = torch.cat([tensor for tensor in x_list], dim=0)
    y = torch.cat([tensor for tensor in y_list], dim=0)
    vec = torch.cat([tensor for tensor in z_list], dim=0)
    # print(x,y,z)
    outputs = net.trainModel(x, vec)
    # print(outputs.argmax(dim=1))
    prelabel.append(outputs.argmax(dim=1).cpu().numpy())
    relabel.append(y.cpu().numpy())
    # print(relabel)
    acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
    n = y.shape[0]
    return acc_sum / n , prelabel, relabel

class MyDataSet(Data.Dataset):
    def __init__(self, data, label, bert_data):
        self.data = data
        self.label = label
        self.bert_data = bert_data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.bert_data[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.31, gamma=6, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Unsupported reduction mode. Use 'mean', 'sum' or 'none'.")

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
import random
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

train_data=torch.load('feature_train.pth')
train_data=train_data[:,:,531:]
bert_train=torch.load('BERT_train.pth')
test_data=torch.load('feature_test.pth')
test_data=test_data[:,:,531:]
bert_test=torch.load('BERT_test.pth')
train_label=torch.load('label_train.pth')
test_label=torch.load('label_test.pth')
bert_train=torch.mean(bert_train,dim=1)
bert_test=torch.mean(bert_test,dim=1)
train_dataset = MyDataSet(train_data, train_label, bert_train)
test_dataset = MyDataSet(test_data, test_label, bert_test)
print(train_data.size())
batch_size = 128
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def calculate_metricsMulti(pred_y, labels):
    test_num = len(labels)
    confusion = confusion_matrix(labels, pred_y)

    # 计算准确度
    ACC = np.sum(np.diag(confusion)) / test_num
    # 计算分类报告
    class_report = classification_report(labels, pred_y, output_dict=True)

    # 组装结果
    metric = {
        'Accuracy': ACC,
        'Precision': class_report['macro avg']['precision'],
        'Recall': class_report['macro avg']['recall'],
        'F1-score': class_report['macro avg']['f1-score']
    }

    return metric

def calculate_metrics_Multi(y_pred, y_true, average='macro'):
    """
    计算多分类问题的各项指标，包括准确率、精确度、召回率、F1 分数、AUC 等，并包含每个标签的独立指标。

    参数：
    - y_true: 真实标签的数组
    - y_pred: 模型预测的标签数组
    - average: 计算平均值的方法，可以是 'micro'、'macro'、'weighted' 或 None，默认为 'macro'

    返回值：
    包含各项指标的字典，以及每个标签的独立指标。
    """
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确度、召回率、F1 分数
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=average)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算 AUC
    unique_labels = np.unique(y_true)
    auc_scores = {}
    for label in unique_labels:
        binary_true = (y_true == label).astype(int)
        binary_pred = (y_pred == label).astype(int)
        auc_scores[label] = roc_auc_score(binary_true, binary_pred)

    # 生成分类报告
    class_report = classification_report(y_true, y_pred, target_names=[str(label) for label in unique_labels], output_dict=True)

    # 构建结果字典
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Confusion Matrix': conf_matrix,
        'AUC Scores': auc_scores,
        'Classification Report': class_report
    }

    return metrics

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def calculate_metrics(y_pred, y_true):

    confusion = confusion_matrix(y_true, y_pred)

    # 计算准确度
    acc = accuracy_score(y_true, y_pred)
    # 计算精确度
    precision = precision_score(y_true, y_pred)
    # 计算召回率
    recall = recall_score(y_true, y_pred)
    # 计算F1-score
    f1 = f1_score(y_true, y_pred)
    # 计算AUC
    auc = roc_auc_score(y_true, y_pred)
    # 计算MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # 组装结果
    metric = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC': auc,
        'MCC': mcc
    }

    return metric

M=0.3
N=1-M
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)
for num_model in range(1):

    net = newModel().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = MultiContrastiveLoss() #对比损失函数

    # criterion_model = FocalLoss()
    criterion_model=nn.CrossEntropyLoss()
    best_acc = 0
    EPOCH = 5

    for epoch in range(EPOCH):
        loss_ls = []
        loss1_ls = []
        loss2_3_ls = []


        t0 = time.time()
        net.train()

        for seq1, seq2, label, label1, label2, pep1, pep2 in train_iter_cont:


            for i in range(len(pep1)):

                # print(pep1)
                # if i == 0:
                #     # print(seq2vec[pep1[i]])
                #     pep1_2vec = torch.tensor(seq2vec[pep1[0]]).to(device)
                #     pep2_2vec = torch.tensor(seq2vec[pep2[0]]).to(device)
                #     # print(pep1_2vec)
                # else:

                #     pep1_2vec = torch.cat((pep1_2vec,torch.tensor(seq2vec[pep1[i]]).to(device)), dim=0)
                #     pep2_2vec = torch.cat((pep2_2vec,torch.tensor(seq2vec[pep2[i]]).to(device)), dim=0)
                #     # print(pep1_2vec)
            pep1=torch.stack(pep1)
            pep2=torch.stack(pep2)
            # print(type(pep1),pep1.size())
            # print(pep1.size(),seq1.size())
            output1 = net(seq1, pep1)
            output2 = net(seq2, pep2)
            output3 = net.trainModel(seq1, pep1)
            output4 = net.trainModel(seq2, pep2)
            loss1 = criterion(output1, output2,label1,label2)
            loss2 = criterion_model(output3, label1)
            loss3 = criterion_model(output4, label2)
            loss = torch.tensor(M).to('cuda')*loss1 + torch.tensor(N).to('cuda')*(loss2 + loss3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            loss1_ls.append(M*loss1.item())
            loss2_3_ls.append(N*(loss2 + loss3).item())
            # output5.extend([output1, output2])
            # label3.extend([label1, label2])
        net.eval()
        with torch.no_grad():
            # print(2)
            train_acc,A_train,B_train = evaluate_accuracy(train_iter, net)
            # print(1)
            test_acc,A,B = evaluate_accuracy(test_iter, net)

            A = [np.concatenate(A)]
            B = [np.concatenate(B)]
            A = np.array(A)
            B = np.array(B)
            A = A.reshape(-1, 1)
            B = B.reshape(-1, 1)

            df1 = pd.DataFrame(A, columns=['prelabel'])
            df2 = pd.DataFrame(B, columns=['realabel'])
            df4 = pd.concat([df1, df2], axis=1)
            acc_sum, n = 0.0, 0
            outputs = []
            for x, y, z in test_iter:
                x, y = x.to(device), y.to(device)
                vec=z
                output = net.trainModel(x, vec)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            pre_pro = outputs[:, 1]
            pre_pro = np.array(pre_pro.cpu().detach().numpy())
            pre_pro = pre_pro.reshape(-1)
            df3 = pd.DataFrame(pre_pro, columns=['pre_pro'])
            df5 = pd.concat([df4, df3], axis=1)
            real1 = df5['realabel']
            pre1 = df5['prelabel']
            metric1 = calculate_metrics_Multi(pre1, real1)


        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        print(results)

        torch.save({"best_acc": best_acc,"metric":metric1, "model": net.state_dict()}, f'{num_model}.pl')
        print(f"best_acc: {best_acc},metric:{metric1}")


