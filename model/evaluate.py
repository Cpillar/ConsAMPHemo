from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import csv
def aaindex(sequences):

  with open('aaindex_feature.csv', 'r') as file:
      csv_reader = csv.reader(file)

      header = next(csv_reader)

      data_dict = {}

      for row in csv_reader:
          name = row[0]  
          values = [float(value) for value in row[1:]]  

          data_dict[name] = values

  # print(data_dict)

    # print(sequence)
  code=[]
  for j in sequences:
      # print(j)
      code = code + data_dict[j]
      # print(len(data_dict[j]))
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
    # seq3=seq[2000:3000]
    # seq4=seq[3000:4000]
    # seq5=seq[4000:]
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
    device = 'cuda'

class newModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 25
        self.emb_dim = 531

        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

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

        # x = self.embedding(x)
        # output = self.transformer_encoder(x).permute(1, 0, 2)
        x=x.float()
        # print(x.size())
        # x=x.reshape(x.shape[0],50,-1)
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
            # saved_tensor_constrative=output.clone().detach().cpu()
            # # torch.save(saved_tensor_constrative,'train_saved_tensor_constrative.pth')
            # # # print(pep.size(),output.size())
            output = torch.cat((output,pep.reshape(output.size(0),-1)),dim=1)
            # output=pep.reshape(output.size(0),-1)

        # output= pep.reshape(-1,1024)

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
    prelabel.append(outputs.argmax(dim=1).cpu().numpy())
    relabel.append(y.cpu().numpy())
    # print(relabel)
    acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
    output=outputs.cpu().numpy()
    soft_outputs=torch.softmax(outputs, dim=1).cpu().numpy()
    n = y.shape[0]
    return acc_sum / n ,output,soft_outputs,prelabel, relabel

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
    def __init__(self, alpha=[0.8, 0.2], gamma=5, reduction='sum'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to('cuda')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

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
train_data=train_data[:,:,:531]
bert_train=torch.load('BERT_train.pth')
test_data=torch.load('feature_test.pth')
test_data=test_data[:,:,:531]
bert_test=torch.load('BERT_test.pth')
train_label=torch.load('label_train.pth')
test_label=torch.load('label_test.pth')
bert_train=torch.mean(bert_train,dim=1)
torch.save(bert_train,'np_saved_bert_train.pth')
bert_test=torch.mean(bert_test,dim=1)
torch.save(bert_test,'combined_saved_bert_test.pth')
train_dataset = MyDataSet(train_data, train_label, bert_train)
test_dataset = MyDataSet(test_data, test_label, bert_test)
print(bert_train.size())
batch_size = 128
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
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

model=newModel().to(device)
pt=torch.load("")
model.load_state_dict(pt['model'])
with torch.no_grad():
    test_acc,outputs,soft_output,prelabel,realabel = evaluate_accuracy(test_iter, model)
print(test_acc)

