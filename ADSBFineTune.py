import os
import sys
# print(sys.path)
# sys.path.insert(0, './models')
import torch
import yaml
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from models.label_smoothing import LabelSmoothingLoss
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from get_dataset import FineTuneDataset_prepared
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random

iteration=50
shot_list=[5,10,15,20,25,30]
pre_tranin=True
label_smooth=False
num_epochs=100
learning_rate=0.001
learning_s=True
# expname='base'
num_class=10
#config
with open("./code/config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
config_ft = config['finetune']
expname='base'


parser = argparse.ArgumentParser(description='PyTorch Complex_test Training')
parser.add_argument('--lr_encoder', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--lr_classifier', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
args = parser.parse_args(args=[])

def train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optimizer_classifier, epoch, device):
    online_network.train()  # 启动训练, 允许更新模型参数
    classifier.train()
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optim_online_network.zero_grad()
        optimizer_classifier.zero_grad()

        # 分类损失反向生成encoder和classifier的梯度
        features = online_network(data)[0]
        # output = F.log_softmax(classifier(features), dim=1)
        output = classifier(features)
        nll_loss_batch = loss_nll(output, target)
        nll_loss_batch.backward()

        optim_online_network.step()
        optimizer_classifier.step()

        nll_loss += nll_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader)


def test(online_network, classifier, test_dataloader, device,config):
    online_network.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    loss=nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            output = classifier(online_network(data)[0])
            # output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

def train_and_test(online_network, classifier, loss_nll, train_dataloader, val_dataloader, optim_online_network, optim_classifier, epochs,  device):
    for epoch in range(1, epochs + 1):
        train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optim_classifier, epoch, device)

def run(train_dataloader, val_dataloader, test_dataloader, epochs,  device,  config,method):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    # online network
    online_network = Encoder_and_projection(**config['network']).to(device)

    # load pre-trained model if defined
    try:
        checkpoints_folder = os.path.join('./result/models', method)
        # load pre-trained parameters
        load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                map_location=torch.device(torch.device(device)))
        filtered_state_dict = {k: v for k, v in load_params['online_network_state_dict'].items() if k not in ['fc.weight', 'fc.bias']}
        filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if 'projetion' not in k}
        online_network.load_state_dict(filtered_state_dict,strict=False)
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    classifier = Classifier()

    if torch.cuda.is_available():
        online_network = online_network.to(device)
        classifier = classifier.to(device)

    loss_nll=nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)

    optim_online_network = torch.optim.Adam(online_network.parameters(), lr=args.lr_classifier, weight_decay=0.0001)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier, weight_decay=0.0001)

    train_and_test(online_network, classifier, loss_nll, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optim_online_network=optim_online_network, optim_classifier=optim_classifier, epochs=epochs,  device=device)
    print("Test_result:")
    test_acc = test(online_network, classifier, test_dataloader, device,config)
    return test_acc ,online_network,classifier

for method1 in  ['SCVNE','SimCLR','BarlowTwins']:#,'SPT','BYOL','AMAE']:
    savepath=f'./result/acc/{method1}/ADSB_result/'
    classifier_save=f'./result/Classifier/ADSB_result/{method1}'
    # def main():
    device = torch.device("cuda:0")
    for k in shot_list:
        test_acc_all = []
        for i in range(iteration):
            print(f"iteration: {i}--------------------------------------------------------")
 
            X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(k)

            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
            train_dataloader = DataLoader(train_dataset, batch_size=config_ft['batch_size'], shuffle=True)

            val_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
            val_dataloader = DataLoader(val_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
            test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)
            # train
            test_acc,online,classer= run(train_dataloader, val_dataloader, test_dataloader, epochs=num_epochs,  device=device,  config=config,method=method1)
            test_acc_all.append(test_acc)
              
        df = pd.DataFrame(test_acc_all)

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        df.to_csv(f"{savepath}class_0_{num_class}_{k}shot_lr{learning_rate}_epoch{num_epochs}_{expname}.csv",index=False)
