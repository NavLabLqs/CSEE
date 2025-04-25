import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
import random
import numpy as np
from models.mlp_head import MLPHead
from models.encoder_and_projection import Encoder_and_projection
from trainer import SCVNETrainer,SCVNETrainer_noavg,SCVNETrainer_noall
from get_dataset import PreTrainDataset_prepared
from torch.utils.data import TensorDataset, DataLoader

print(torch.__version__)

def main():
    config = yaml.load(open("./config/config.yaml", "r",encoding='utf-8'), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    X_train_ul, Y_train_ul = PreTrainDataset_prepared()
    train_dataset = TensorDataset(torch.Tensor(X_train_ul), torch.Tensor(Y_train_ul))

    # online network
    online_network = Encoder_and_projection(**config['network']).to(device)

    # target encoder
    target_network = Encoder_and_projection(**config['network']).to(device)

    optimizer = torch.optim.Adam(list(online_network.parameters()),
                                **config['optimizer']['params'])

    trainer = SCVNETrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
