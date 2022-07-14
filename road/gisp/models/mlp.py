import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, n_layers, layer_factor, 
            act_fn='relu', dropout_p=0.0):
        super(MLP, self).__init__()
        n_h = int(n_in * layer_factor)
        act_fn_class = utils.get_activation(act_fn)
        # build the net
        net = []
        for i in range(n_layers):
            if i == 0:
                net.append(torch.nn.Linear(n_in, n_h))
            else:
                net.append(torch.nn.Linear(n_h, n_h))
            net.append(act_fn_class())
            net.append(torch.nn.BatchNorm1d(n_h))
            if dropout_p > 0:
                net.append(torch.nn.Dropout(p=dropout_p))

        net.append(torch.nn.Linear(n_h, n_out))
        # convert to sequential
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        x_out = self.net(x)
        return x_out

class PNet_FC(nn.Module):
    def __init__(self, n_features, n_layers, n_classes):
        super(PNet_FC, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.net = nn.Sequential(
                nn.Linear(self.n_features, self.n_layers), 
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d((self.n_layers)), 
                nn.Linear(self.n_layers, self.n_layers),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d((self.n_layers)), 
                nn.Linear(self.n_layers, n_classes),
                )
    def forward(self, x_in):
        # flatten the input
        x = x_in.view(x_in.shape[0], -1)
        x = self.net(x)
        # return output
        return x


