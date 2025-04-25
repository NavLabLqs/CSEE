import os
import sys
print(sys.path)
sys.path.insert(0, './models')
import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F
from models.complexcnn import ComplexConv, ComplexConv_trans
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import *
import numpy as np
import random
import yaml
from torch.optim.lr_scheduler import StepLR
import pandas as pd

class CVCNN(nn.Module):
    def __init__(self, num_class,embed_len,*args, **kwargs):
        super(CVCNN, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embed_len, 1024)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        # x=self.fam(x)
        x=F.relu(self.fc1(x))

    
        x=self.fc2(x)
        return x
    
#----------------------config-----------------------------
iteration=50
shot_list=[25]
pre_tranin=True
label_smooth=False
num_epochs=400
learning_rate=0.01
learning_s=True
dataset='LORA'
method='SCVNE'
expname='base'
for dataset in ['LORA']:
    if dataset=='LORA':
        num_class=10
    if dataset=='AIS':
        num_class=7
    #---------------------------------------------------------
    if dataset=='LORA':
        x = np.load(f"./dataset/LORA/Train/LORA_SF8_data.npy")
        y = np.load(f"./dataset/LORA/Train/LORA_SF8_label.npy")
        x = x.transpose(0, 2, 1)
        x_test = np.load(f"./dataset/LORA/Test/LORA_SF8_data_te.npy")
        y_test = np.load(f"./dataset/LORA/Test/LORA_SF8_label_te.npy")
        x_test = x_test.transpose(0, 2, 1)
    elif dataset == 'AIS':
        x = np.load(f"./dataset/AIS/Train/AIS_train.npy")
        y = np.load(f"./dataset/AIS/Train/AIS_label_train.npy")
        x = x.transpose(0, 2, 1)
        x_test = np.load(f"./dataset/AIS/Test/AIS_test.npy")
        y_test = np.load(f"./dataset/AIS/Test/AIS_label_test.npy")
        x_test = x_test.transpose(0, 2, 1)
    for method in ['SCVNE','SimCLR','BarlowTwins']:#,'SPT','BYOL','AMAE']:
        savepath=f'./result/acc/{method}/{dataset}_result/'
        print(f'pre_tranin:{pre_tranin},dataset:{dataset},method:{method}')
        for k in shot_list:

            train_losses = []
            test_accuracy = []
            for j in range(iteration):

                print(f'shot:{k},iteration:{j}')
                # print('------------------------------------------------------')
                finetune_index_shot = []
                for i in range(num_class):
                    index_classi = [index for index, value in enumerate(y) if value == i]
                    finetune_index_shot += random.sample(index_classi, k)
                X_train, X_test, Y_train, Y_test=x[finetune_index_shot], x_test, y[finetune_index_shot], y_test
                Y_train = Y_train.astype(np.uint8).reshape(-1)
                Y_test = Y_test.astype(np.uint8).reshape(-1)

                max_value = X_train.max()
                min_value = X_train.min()

                X_train = (X_train - min_value) / (max_value - min_value)
                X_test = (X_test - min_value) / (max_value - min_value)

                device='cuda'
                train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(Y_train,dtype=torch.long))
                test_dataset=TensorDataset(torch.Tensor(X_test), torch.tensor(Y_test,dtype=torch.long))

                if dataset=='AIS':
                    model=CVCNN(num_class,896).to(device)
                elif dataset=='LORA':
                    model=CVCNN(num_class,768).to(device)
                # config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
                with open("./code/config/config.yaml", "r", encoding="utf-8") as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
                if pre_tranin:
                    try:
                        if method=='SCVNE' and dataset=='AIS':
                            checkpoints_folder = os.path.join('./result/models', f'{method}','AIS')
                        elif method=='SCVNE' and dataset=='LORA':
                            checkpoints_folder = os.path.join('./result/models', f'{method}','LORA')    
                        else:
                            checkpoints_folder = os.path.join('./result/models', f'{method}')

                        # load pre-trained parameters
                        load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                                    map_location=torch.device(torch.device(device)))
                        

                        filtered_state_dict = {k: v for k, v in load_params['online_network_state_dict'].items() if k not in ['fc.weight', 'fc.bias']}
                        filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if 'projetion' not in k}
                        model.load_state_dict(filtered_state_dict,strict=False)
                    except FileNotFoundError:
                        print("Pre-trained weights not found. Training from scratch.")

                criterion = nn.CrossEntropyLoss()

                optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

                for epoch in range(num_epochs):

                    loop = tqdm((train_loader), total = len(train_loader),ncols=100,colour='green',leave=False)

                    model.train()  # 设置模型为训练模式

                    for data, labels in loop:
                        data, labels = data.to(device), labels.to(device)
                        # 前向传播
                        outputs = model(data)
                        # 计算损失
                        loss = criterion(outputs, labels)
                        _,predictions = outputs.max(1)
                        num_correct = (predictions == labels).sum()
                        running_train_acc = float(num_correct) / float(data.shape[0])
                        # 反向传播和优化
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                        loop.set_postfix(loss = f'{loss.item():.3f}',acc = f'{running_train_acc:.4f}')
                    # 学习率调整策略
                    if learning_s:
                        if epoch == 200:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = 0.001  
                            print('lr:0.001')     
                        # print('------------------')    
                    
                model.eval()  # 设置模型为评估模式
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, labels in test_loader:
                        data, labels = data.to(device), labels.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total
                # torch.save(model,'./model_CVCNN.pth')
                train_losses.append(loss.item())
                test_accuracy.append(accuracy)
                print(f'Accuracy_test: {accuracy:.4f}')
            df = pd.DataFrame(test_accuracy)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            df.to_csv(f"{savepath}class_0_{num_class}_{k}shot_lr{learning_rate}_epoch{epoch}_{expname}.csv",index=False)

