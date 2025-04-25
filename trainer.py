import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from utils import _create_model_training_folder, Data_augment
from torch.autograd import Variable

class CSEETrainer:
    def __init__(self, online_network, target_network,optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.beta=params['beta']
        self.rho = params['rho']
        self.writer = SummaryWriter(f"runs/CSEE")
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.final_dim = params['final_dim']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    def CSEE_loss(self,C,batch_view_1,batch_view_2):
        online_from_view_1 = self.online_network(batch_view_1)[1]
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_2)[1]
        z_1 = torch.nn.functional.normalize(online_from_view_1, dim = -1)
        z_2 = torch.nn.functional.normalize(targets_to_view_2, dim = -1)
        B = torch.mm(z_1.T, z_1)/z_1.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        loss = - (z_1 * z_2).sum(dim=1).mean() - self.rho * self.get_vne(C)
        return loss, C

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def get_vne(self,H):
        Z=H
        sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
        eig_val = sing_val 
        return - (eig_val * torch.log(eig_val)).nansum()
    
    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, awgn=False, add_noise=False, slice=False, isAdvAug=True)

        C_prev = Variable(torch.zeros(self.final_dim,self.final_dim), requires_grad=True).to(self.device)
        C_prev = C_prev.detach()

        for epoch_counter in range(self.max_epochs):
            print(f'Epoch={epoch_counter}')
            loss_epoch = 0
            loss_min = 10000000

            for batch_view, _ in train_loader:
                batch_view = batch_view.to(self.device)
                batch_view_1 = Augment(batch_view, copy.deepcopy(self.online_network))
                batch_view_2 = Augment(batch_view, copy.deepcopy(self.online_network))

                loss,C = self.CSEE(C_prev,batch_view_1, batch_view_2)
                C_prev = C.detach()
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch += loss.item()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            loss_epoch /= len(train_loader)
            if loss_epoch <= loss_min:
                self.save_model(os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = loss_epoch
            self.writer.add_scalar('loss_epoch', loss_epoch, global_step=epoch_counter)
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
    