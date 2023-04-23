import os
import gc
import cv2
import torch
import random
import string
import tifffile
import numpy as np 
import pandas as pd 
import torch.nn as nn
from random import randint
from torchvision import models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")
gc.enable()


debug = False
generate_new = False
train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv").head(10 if debug else 329)
test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
dirs = ["../input/mayo-clinic-strip-ai/train/", "../input/mayo-clinic-strip-ai/test/"] # this is the folder of .tif files


#3
max_count = max(train_df.label.value_counts())
#print(max_count)
for label in ['CE','LAA']:
    df = train_df.loc[train_df.label == label] #train_df.label == label构成了一个布尔序列，把train_df中的对应列选出来
    #print(df)
    while(train_df.label.value_counts()[label] < max_count): #
        train_df = pd.concat([train_df, df.head(max_count - train_df.label.value_counts()[label])], axis = 0)
#these codes are for making the size of two sets equally 
#train_df.label.value_counts()

class ImgDataset(Dataset):
    def __init__(self, df):
        self.df = df 
        self.train = 'label' in df.columns    #self.train是‘是不是训练集’的意思。应该命名为is_train
    def __len__(self): return len(self.df)    
    def __getitem__(self, index):
        if(1): paths = ["./test/", "./train/"] #generate_new 我现在只做output里缩小版的图片
        image = cv2.imread(paths[self.train] + self.df.iloc[index].image_id + ".jpg")

        image = cv2.resize(image, (512, 512)).transpose(2, 0, 1)
        label = None
        if(self.train): label = {"CE" : 0, "LAA": 1}[self.df.iloc[index].label]
        return image, label

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    tran_acc_list=[]
    valid_acc_list=[]
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.cuda()       
        flag=0
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
               
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                images = item[0].cuda().float()
                classes = item[1].cuda().long()
                optimizer.zero_grad()                
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)
                    loss = criterion(output, classes)
                    _, preds = torch.max(output, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * len(output)
                    epoch_acc += torch.sum(preds == classes.data)                    
            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            if phase=='train':
                tran_acc_list.append(epoch_acc.item())
            else:valid_acc_list.append(epoch_acc.item())
            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')    
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 512, 512))
            traced.save('model.pth')
            best_acc = epoch_acc
    return tran_acc_list,valid_acc_list

#model = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b4")
model = EfficientNet.from_name("efficientnet-b0")
model.set_swish(memory_efficient = False)
checkpoint = torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth')
model.load_state_dict(checkpoint)
model.set_swish(memory_efficient = False)

train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify = train_df.label)

train_loader = DataLoader(ImgDataset(train), batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(ImgDataset(val), batch_size=1, shuffle=True, num_workers=1)
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_acc,test_acc=train_model(model, dataloaders_dict, criterion, optimizer, 8)

n_splits = 4
from sklearn.model_selection import KFold
from torch.utils.data import Subset
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

valid_acc_set=[]
train_acc_set=[]
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    if fold>1:continue
    train_set = Subset(ImgDataset(train_df), train_idx)
    val_set = Subset(ImgDataset(train_df), val_idx)
    
    model = EfficientNet.from_name("efficientnet-b1")
    model.set_swish(memory_efficient = False)
    checkpoint = torch.load('../input/efficientnet-pytorch/efficientnet-b1-dbc7070a.pth')
    model.load_state_dict(checkpoint)
    model.set_swish(memory_efficient = False)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=1)
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_acc,valid_acc=train_model(model, dataloaders_dict, criterion, optimizer, 6)
    valid_acc_set.append(valid_acc)
    train_acc_set.append(train_acc)


import matplotlib.pyplot as plt

# Convert tensors to numbers using item()
train_acc=[]   #

# Plot the training and test accuracy
plt.plot(range(1,7), train_acc_set[1], label='training accuracy')
plt.plot(range(1,7), valid_acc_set[0],label='validation accuracy')
plt.ylim(0,1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print(train_acc)


sum=0
for i in valid_acc_set:
    sum+=max(i)
avg=sum/len(valid_acc_set)
print(avg)
