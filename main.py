import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

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

#模型类
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear=nn.Linear(in_features=30,out_features=1,bias=True)
        self.sigmod=nn.Sigmoid()
 
    def forward(self,x):
        x=self.linear(x)
        x=self.sigmod(x)
        return x
       
csv_path='/home/rpi/midterm_project/origin_breast_cancer_data.csv'
csv_data=pd.read_csv(csv_path)
dataset_raw=dataset(csv_data)
# ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'],'diagnosis')
train_size = int(0.70 * len(dataset_raw))
test_size = len(dataset_raw) - train_size
torch.manual_seed(0)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset_raw, [train_size, test_size])
data_loader = DataLoader(train_dataset,shuffle=True,batch_size=16,drop_last=True)
validation_loader = DataLoader(validation_dataset,shuffle=True,batch_size=5,drop_last=True)


model=Model()
epochs = 600
# learning_rate = 0.00002
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
criterion = nn.BCELoss()
loss_sum=[]

for epoch in range(epochs):
    loss_sum=[]
    for step, (features, labels) in enumerate(data_loader): 
        out = model(features.to(torch.float32))
        labels=labels.unsqueeze(-1).to(torch.float32)
        loss = criterion(out, labels)
        loss_sum.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



TP=0
TN=0
FP=0
FN=0

for step, (features, labels) in enumerate(data_loader): 
    output=model(features.to(torch.float32))
    labels=labels.unsqueeze(-1).to(torch.float32)
    for i in range(len(labels)):
        if output[i]>0.5 and labels[i]>0.5:
            TP+=1
        elif output[i]<0.5 and labels[i]>0.5:
            FN+=1
        elif output[i]>0.5 and labels[i]<0.5:
            FP+=1
        elif output[i]<0.5 and labels[i]<0.5:
            TN+=1
print(TP,TN,FP,FN)
precession_train=TP/(TP+FP)
print(F'precesion for train is {precession_train}')
recall_train=TP/(TP+FN)    
print(F'recall for tarin is {recall_train}')
F1_score=2*precession_train*recall_train/(precession_train+recall_train)
print(F'F1_score for train is {F1_score}')

TP=0
TN=0
FP=0
FN=0


for step, (features, labels) in enumerate(validation_loader): 
    output=model(features.to(torch.float32))
    labels=labels.unsqueeze(-1).to(torch.float32)
    for i in range(len(labels)):
        if output[i]>0.5 and labels[i]>0.5:
            TP+=1
        elif output[i]<0.5 and labels[i]>0.5:
            FN+=1
        elif output[i]>0.5 and labels[i]<0.5:
            FP+=1
        elif output[i]<0.5 and labels[i]<0.5:
            TN+=1
print(TP,TN,FP,FN)
precession_train=TP/(TP+FP)
print(F'precesion for validation is {precession_train}')
recall_train=TP/(TP+FN)    
print(F'recall for validation is {recall_train}')
F1_score=2*precession_train*recall_train/(precession_train+recall_train)
print(F'F1_score for validation is {F1_score}')














