import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
#数据集
class dataset(Dataset):
    def __init__(self, csv_data):
        # 输入的为第三列到最后一列一共30个features
        X = csv_data[csv_data.columns[2:-1]].values
        # labels为第二列的M：恶性 B：良性
        Y = csv_data[csv_data.columns[1]].values
        self.dataset = torch.from_numpy(X)
        tagetset=torch.zeros(len(Y))
        for id,items in enumerate(Y):
            # 恶性为1，良性为0。
            tagetset[id]=1 if items=='M' else 0
        self.targetset =tagetset

    def __getitem__(self, index):
        return self.dataset[index], self.targetset[index]

    def __len__(self):
        return self.dataset.size(0)

def MinMaxScaler(Mat):
    a,b=torch.max(Mat, dim=0, keepdim=True)
    c,d=torch.min(Mat, dim=0, keepdim=True)
    e=(Mat-c)/(a-c)
    return e

    


global csvpath
csvpath= ['/home/rpi/midterm_project/origin_breast_cancer_data.csv','/home/rpi/midterm_project/breast_cancer_data_357B_100M.csv']

def getdata(choice):
    csv_path=csvpath[choice]
    csv_data=pd.read_csv(csv_path)
    dataset_raw=dataset(csv_data)
    torch.manual_seed(0)
    train_size = int(0.70 * len(dataset_raw))
    test_size = len(dataset_raw) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset_raw, [train_size, test_size])
    data_loader = DataLoader(train_dataset,shuffle=True)
    validation_loader = DataLoader(validation_dataset,shuffle=True)

    x_train =[]
    y_train =[]
    for id,(features, labels) in enumerate(data_loader): 
        if id == 0:
            x_train = features
        else:
            x_train = torch.cat((x_train, features), dim=0)
        y_train.append(labels.item())
    y_train=torch.tensor(y_train).to(torch.int16)
    x_train=MinMaxScaler(x_train)


    x_test =[]
    y_test =[]
    for id,(features, labels) in enumerate(validation_loader): 
        if id == 0:
            x_test = features
        else:
            x_test = torch.cat((x_test, features), dim=0)
        y_test.append(labels.item())
    y_test=torch.tensor(y_test).to(torch.int16)
    x_test=MinMaxScaler(x_test)
    return x_train,y_train,x_test,y_test






def KNN(train_x, train_y, test_x, test_y, k):
    m = test_x.size(0)
    n = train_x.size(0)
    xx = (test_x**2).sum(dim=1,keepdim=True).expand(m, n)
    yy = (train_x**2).sum(dim=1, keepdim=True).expand(n, m).transpose(0,1)

    dist_mat = xx + yy - 2*test_x.matmul(train_x.transpose(0,1))
    mink_idxs = dist_mat.argsort(dim=-1)
    res = []
    for idxs in mink_idxs:
        res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())
    
    assert len(res) == len(test_y)
    acc=accuracy_score(test_y, res)
    return acc


for kenal in range(1,30):
    print('k=',kenal)
    acc=[]
    for i in range(30):
        x_train,y_train,x_test,y_test=getdata(0)
        acc.append(KNN(x_train,y_train,x_test,y_test,kenal))
    print('avg:',sum(acc)/len(acc))
    print("min:",min(acc))