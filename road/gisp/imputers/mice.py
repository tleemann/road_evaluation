import pdb
import os
import argparse
import time
import json
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision

#from autoimpute.imputations import MultipleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import utils
import models.autoencoder
import data


class GNet_MI(torch.nn.Module):
    def __init__(self, imp):
        super(GNet_MI, self).__init__()
        self.imp = imp

    def forward(self, x_m):
        device = x_m.device
        mask = 1.0 - torch.isnan(x_m)
        x_m = pd.DataFrame(x_m.cpu().numpy())
        #x_i = next(self.imp.transform(x_m))[1].values
        #x_m = pd.DataFrame(x_m.cpu().numpy())
        x_i = self.imp.transform(x_m)
        x_i = torch.tensor(x_i, dtype=torch.float32).to(device)
        return x_i


class Imputer:
    def __init__(self, sample_limit=5000):
        self.model = None
        self.sample_limit = sample_limit

    def __call__(self, x):
        return self.model(x)

    def to(self, x):
        self.model = self.model.to(x)
        return self
    
    def save(path):
        torch.save(self, path)
    
    def load(path):
        obj = torch.load(path, map_location='cpu')
        self.__dict__.update(obj.__dict__)
        return self
  
    def train(self, train_loader, test_loader, mfv, args):
        run_trace = {}
        n_features = mfv.view(-1).size()[0]
        shape_orig = mfv.size()

        # make a training batch
        inds_sel = np.random.permutation(len(train_loader))[:self.sample_limit]
        features_trn = []
        for ind_batch, ((x,x_m),y) in enumerate(train_loader):
            if ind_batch in inds_sel:
                features_trn.append(x_m.cpu().numpy())
        features_trn = np.vstack(features_trn)
        
        # make an MI model
        #imp = MultipleImputer()
        imp = IterativeImputer(sample_posterior=True, random_state=None, 
                n_nearest_features=min(100,features_trn.shape[1]))
        #imp.fit(pd.DataFrame(features_trn))
        imp.fit(features_trn)
        
        # make the fake pytorch model
        self.model = GNet_MI(imp)
        




