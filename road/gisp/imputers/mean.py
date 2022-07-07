import pdb
import os
import argparse
import time
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision

import utils
import models.autoencoder
import data


class GNet_Const(torch.nn.Module):
    def __init__(self, val):
        super(GNet_Const, self).__init__()
        self.val = val

    def forward(self, x_m):
        device = x_m.device
        mask = 1.0 - torch.isnan(x_m)
        x = torch.where(mask.bool(), x_m, torch.tensor(self.val, device=device))
        return x


class Imputer:
    def __init__(self):
        self.model = None

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
        if len(args.device.split(',')) == 1:
            device = torch.device(args.device)
            if args.device[:4] == 'cuda':
                torch.cuda.set_device(int(args.device.split(':')[-1]))
        else:
            raise NotImplementedError
      
        model_gnet = GNet_Const(0.0).to(device)

        iter_trn = 0
        epoch_trn = 0
        run_trace['TRN_ITER'] = [0]
        run_trace['TRN_EPOCH'] = [0]
        run_trace['TRN_LOSS_G'] = [np.nan]
        run_trace['TRN_LOSS_D'] = [np.nan]
        run_trace['VAL_ITER'] = [0]
        run_trace['VAL_EPOCH'] = [0]
        run_trace['VAL_LOSS_G'] = [np.nan]
        run_trace['VAL_LOSS_D'] = [np.nan]
        run_trace['VAL_FID'] = [] 
        run_trace['VAL_MSE'] = []   
        # visual evaluation
        if not args.no_vis:
            utils.eval_visual(test_loader, model_gnet, mfv, device, 
                    iter_trn, epoch_trn, args)
        # FID, MSE scores
        mse_score = utils.calc_mse_score(test_loader, mfv, model_g=model_gnet, 
                dataset=args.dataset, device=device)
        if args.dataset not in data.DENSE_DATASETS:
            fid_score = utils.calc_fid_score(test_loader, mfv, model_g=model_gnet, 
                    dataset=args.dataset, device=device)
        else:
            fid_score = np.nan

        # update trace
        run_trace['VAL_FID'].append(fid_score)
        run_trace['VAL_MSE'].append(mse_score)
        utils.dump_trace(args.result_dir, run_trace)

        self.model = model_gnet.cpu()
        # preprate the model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        #torch.save(self, 
        #        args.data_dir + '/model/{}.pt'.format(args.conf_hash))




