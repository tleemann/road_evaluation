import sys
import pdb
import os
import argparse
import time
import json
import copy

import numpy as np
from matplotlib.patches import Rectangle
import pylab as plt
from tqdm import tqdm
import torch
from torch.autograd import grad
import torchvision

import utils
import models.misgan
import data




class CriticUpdater:
    def __init__(self, critic, critic_optimizer, eps, ones, gp_lambda, args):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.gp_lambda = gp_lambda

    def __call__(self, real, fake):
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()


def mask_data(data, mask, tau=0.5):
    return mask * data + (1 - mask) * tau

def plot_grid(ax, image, bbox=None, gap=1, gap_value=1, nrow=4, ncol=8,
              title=None):
    image = image.cpu().numpy()#.squeeze(1)
    LEN = image.shape[-1]
    n_channels = image.shape[1]
    grid = np.empty((3, nrow * (LEN + gap) - gap, ncol * (LEN + gap) - gap))
    grid.fill(gap_value)

    for i, x in enumerate(image):
        if i >= nrow * ncol:
            break
        p0 = (i // ncol) * (LEN + gap)
        p1 = (i % ncol) * (LEN + gap)
        grid[:, p0:(p0 + LEN), p1:(p1 + LEN)] = x

    ax.set_axis_off()
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 0, 1)
    ax.imshow(grid, interpolation='none', aspect='equal')

    if bbox:
        nplot = min(len(image), nrow * ncol)
        for i in range(nplot):
            d0, d1, d0_len, d1_len = bbox[i]
            p0 = (i // ncol) * (LEN + gap)
            p1 = (i % ncol) * (LEN + gap)
            offset = np.array([p1 + d1, p0 + d0]) - .5
            ax.add_patch(Rectangle(
                offset, d1_len, d0_len, lw=1.5, edgecolor='red', fill=False))
            
    if title:
        ax.set_title(title)


class GNetWrapper:
    def __init__(self, model_impu, impu_noise, mfv):
        self.model_impu = model_impu
        self.impu_noise = impu_noise
        self.mfv = mfv

    def __call__(self, x_m):
        # copy x_m
        x_m = copy.deepcopy(x_m)
        self.impu_noise.uniform_()
        x_m += self.mfv
        mask = 1 - torch.isnan(x_m)
        if len(mask.shape) == 4:
            real_mask = torch.unsqueeze(mask[:,0,:,:], 1).float()
        else:
            real_mask = mask.float()
        real_data = torch.where(mask.bool(), x_m,  torch.tensor(0.0, device=x_m.device))
        self.impu_noise.uniform_()
        x_g = self.model_impu(real_data, real_mask, self.impu_noise)
        x_g -= self.mfv
        return x_g

    def to(self, device):
        self.model_impu = self.model_impu.to(device)
        self.impu_noise = self.impu_noise.to(device)
        self.mfv = self.mfv.to(device)
        return self

    def eval(self):
        self.model_impu.eval()

    def parameters(self):
        return self.model_impu.parameters()


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
       
        # default settings
        nz = 128   # dimensionality of the latent code
        n_critic = 5
        gp_lambda = 10
        alpha = .1
        beta = 0.1
        tau = 0.5
        lrate = 1e-4
        imputer_lrate = 2e-4
        # define models
        if args.dataset == 'mnist':
            #tau = 0.0
            model_data_gen = models.misgan.ConvDataGeneratorMnist()
            model_mask_gen = models.misgan.ConvMaskGeneratorMnist()
            model_data_critic = models.misgan.ConvCriticMnist()
            model_mask_critic = models.misgan.ConvCriticMnist()
            model_imputer_gen = models.misgan.ComplementImputerMnist()
            model_imputer_critic = models.misgan.ConvCriticMnist()
        elif args.dataset == 'cifar10':
            model_data_gen = models.misgan.ConvDataGeneratorCifar()
            model_mask_gen = models.misgan.ConvMaskGeneratorCifar()
            model_data_critic = models.misgan.ConvCriticCifar(n_channels=3)
            model_mask_critic = models.misgan.ConvCriticCifar(n_channels=1)
            model_imputer_gen = models.misgan.UNetImputer()
            model_imputer_critic = models.misgan.ConvCriticCifar(n_channels=3)
        elif args.dataset == 'celeba':
            model_data_gen = models.misgan.ConvDataGeneratorCeleb()
            model_mask_gen = models.misgan.ConvMaskGeneratorCeleb()
            model_data_critic = models.misgan.ConvCriticCeleb(n_channels=3)
            model_mask_critic = models.misgan.ConvCriticCeleb(n_channels=1)
            model_imputer_gen = models.misgan.UNetImputer()
            model_imputer_critic = models.misgan.ConvCriticCeleb(n_channels=3)
        elif args.dataset in data.DENSE_DATASETS:
            nz = n_features // 8
            model_data_gen = models.misgan.FCDataGenerator(n_features)
            model_mask_gen = models.misgan.FCMaskGenerator(n_features)
            model_data_critic = models.misgan.FCCritic(n_features)
            model_mask_critic = models.misgan.FCCritic(n_features)
            model_imputer_gen = models.misgan.FCImputer(n_features)
            model_imputer_critic = models.misgan.FCCritic(n_features)
        else:
            raise NotImplementedError
        
        if len(mfv.shape) == 3:
            eps = torch.FloatTensor(args.batch_size, 1, 1, 1).to(device)
        else:
            eps = torch.FloatTensor(args.batch_size, 1).to(device)
        ones = torch.ones(args.batch_size).to(device)

        model_data_gen = model_data_gen.to(device)
        model_mask_gen = model_mask_gen.to(device)
        model_data_critic = model_data_critic.to(device)
        model_mask_critic = model_mask_critic.to(device)
        model_imputer_gen = model_imputer_gen.to(device)
        model_imputer_critic = model_imputer_critic.to(device)
        data_noise = torch.empty(args.batch_size, nz, device=device)
        mask_noise = torch.empty(args.batch_size, nz, device=device)
        impu_noise = torch.empty(args.batch_size, *list(mfv.shape), device=device)
        # define optimizers
        data_gen_optimizer = torch.optim.Adam(
                model_data_gen.parameters(), lr=lrate, betas=(.5, .9))
        mask_gen_optimizer = torch.optim.Adam(
                model_mask_gen.parameters(), lr=lrate, betas=(.5, .9))
        data_critic_optimizer = torch.optim.Adam(
                model_data_critic.parameters(), lr=lrate, betas=(.5, .9))
        mask_critic_optimizer = torch.optim.Adam(
                model_mask_critic.parameters(), lr=lrate, betas=(.5, .9))
        imputer_optimizer = torch.optim.Adam(
                    model_imputer_gen.parameters(), lr=imputer_lrate, betas=(.5, .9))
        impu_critic_optimizer = torch.optim.Adam(
                    model_imputer_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))
        update_data_critic = CriticUpdater(
                model_data_critic, data_critic_optimizer, eps, ones, gp_lambda, args) 
        update_mask_critic = CriticUpdater(
                model_mask_critic, mask_critic_optimizer, eps, ones, gp_lambda, args)
        update_impu_critic = CriticUpdater(
                    model_imputer_critic, impu_critic_optimizer, eps, ones, gp_lambda, args)
        # multiphase obj parsing: mse_advF_advFV
        if '_' in args.objective:
            phases_objective = {}
            n_phases = len(args.objective.split('_'))
            for ind, obj in enumerate(args.objective.split('_')):
                phases_objective[ind*int(args.epoches/n_phases)] = obj
        else:
            phases_objective = {0:args.objective}

        # train
        run_trace['TRN_ITER'] = []
        run_trace['TRN_EPOCH'] = []
        run_trace['TRN_LOSS_G'] = []
        run_trace['TRN_LOSS_D'] = []
        run_trace['VAL_ITER'] = []
        run_trace['VAL_EPOCH'] = []
        run_trace['VAL_LOSS_G'] = []
        run_trace['VAL_LOSS_D'] = []
        run_trace['VAL_FID'] = [] 
        run_trace['VAL_MSE'] = []  
        run_trace['VAL_R'] = []  
        best_model = None
        best_dist = np.inf
        iter_trn = 0
        n_batches = len(train_loader)
        n_iters = args.epoches * n_batches
        """
        pbar = tqdm(total=n_iters)
        for epoch_trn in range(args.epoches):
            # multiphase obj support
            if epoch_trn in phases_objective:
                args.objective = phases_objective[epoch_trn]
            for ind_batch, ((x,x_m),y) in enumerate(train_loader):
                iter_trn += 1
                # get a batch
                #x = x.to(device) - mfv
                x_m = x_m.to(device)
                mask = 1 - torch.isnan(x_m)
                if len(mask.shape) == 4:
                    real_mask = torch.unsqueeze(mask[:,0,:,:], 1).float()
                else:
                    real_mask = mask.float()
                real_data = torch.where(mask.byte(), x_m, 
                        torch.tensor(0.0, device=device))

                # Update discriminators' parameters
                data_noise.normal_()
                mask_noise.normal_()

                fake_data = model_data_gen(data_noise)
                fake_mask = model_mask_gen(mask_noise)

                masked_fake_data = mask_data(fake_data, fake_mask)
                masked_real_data = mask_data(real_data, real_mask)

                update_data_critic(masked_real_data, masked_fake_data)
                update_mask_critic(real_mask, fake_mask)

                if iter_trn % n_critic == 0:
                    # Update generators' parameters
                    for p in model_data_critic.parameters():
                        p.requires_grad_(False)
                    for p in model_mask_critic.parameters():
                        p.requires_grad_(False)

                    model_data_gen.zero_grad()
                    model_mask_gen.zero_grad()

                    data_noise.normal_()
                    mask_noise.normal_()

                    fake_data = model_data_gen(data_noise)
                    fake_mask = model_mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask)

                    data_loss = -model_data_critic(masked_fake_data).mean()
                    data_loss.backward(retain_graph=True)
                    data_gen_optimizer.step()

                    mask_loss = -model_mask_critic(fake_mask).mean()
                    (mask_loss + data_loss * alpha).backward()
                    mask_gen_optimizer.step()

                    for p in model_data_critic.parameters():
                        p.requires_grad_(True)
                    for p in model_mask_critic.parameters():
                        p.requires_grad_(True)

                if iter_trn % (max(n_iters//10000,1)) == 0:
                    run_trace['TRN_ITER'].append(iter_trn)
                    run_trace['TRN_EPOCH'].append(epoch_trn)
                    #run_trace['TRN_LOSS_G'].append(loss_g.item())
                    #run_trace['TRN_LOSS_D'].append(np.nan)
                    #pbar.set_description('Training, epoch={}, loss g={:1.2e}'.format(
                    #    epoch_trn, loss_g.item()))
                    pbar.set_description('Training, Phase 1, epoch={}'.format(epoch_trn))
                pbar.update()
            
            # TODO: validation
            if epoch_trn % max(int(args.eval_freq*args.epoches),1) == 0:
                model_data_gen.eval()
                model_mask_gen.eval()

                with torch.no_grad():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
                    
                    data_noise.normal_()
                    data_samples = model_data_gen(data_noise)
                    plot_grid(ax1, data_samples, title='generated complete data')
                    
                    mask_noise.normal_()
                    mask_samples = model_mask_gen(mask_noise)
                    plot_grid(ax2, mask_samples, title='generated masks')
                    
                    plt.savefig(
                            args.result_dir+'images_p1_obj{}_epoch{}_iter{}.png'.format(
                                args.objective, epoch_trn, iter_trn))
                    plt.close(fig)

                model_data_gen.train()
                model_mask_gen.train()
        pbar.close()
        """
        ## PHASE 2 
        run_trace2 = {}
        run_trace2['TRN_ITER'] = []
        run_trace2['TRN_EPOCH'] = []
        run_trace2['TRN_LOSS_G'] = []
        run_trace2['TRN_LOSS_D'] = []
        run_trace2['VAL_ITER'] = []
        run_trace2['VAL_EPOCH'] = []
        run_trace2['VAL_LOSS_G'] = []
        run_trace2['VAL_LOSS_D'] = []
        run_trace2['VAL_FID'] = [] 
        run_trace2['VAL_MSE'] = []  
        run_trace2['VAL_R'] = []  
        pbar = tqdm(total=n_iters)
        for epoch_trn in range(args.epoches):
            # multiphase obj support
            if epoch_trn in phases_objective:
                args.objective = phases_objective[epoch_trn]
            for ind_batch, ((x,x_m),y) in enumerate(train_loader):
                iter_trn += 1
                # wait for at least 10gb memory
                utils.wait_for_memory(10)

                # get a batch
                #x = x.to(device) - mfv
                x_m = x_m.to(device)
                mask = 1 - torch.isnan(x_m)
                if len(mask.shape) == 4:
                    real_mask = torch.unsqueeze(mask[:,0,:,:], 1).float()
                else:
                    real_mask = mask.float()
                real_data = torch.where(mask.bool(), x_m, 
                        torch.tensor(0.0, device=device))

                # Update discriminators' parameters
                data_noise.normal_()
                mask_noise.normal_()
                impu_noise.uniform_()

                fake_data = model_data_gen(data_noise)
                fake_mask = model_mask_gen(mask_noise)

                masked_fake_data = mask_data(fake_data, fake_mask, tau)
                masked_real_data = mask_data(real_data, real_mask, tau)
                
                imputed_data = model_imputer_gen(real_data, real_mask, impu_noise)
                masked_imputed_data = mask_data(real_data, real_mask, imputed_data)
                
                update_data_critic(masked_real_data, masked_fake_data)
                update_mask_critic(real_mask, fake_mask)
                update_impu_critic(fake_data, masked_imputed_data)

                if iter_trn % n_critic == 0:
                    # Update generators' parameters
                    for p in model_data_critic.parameters():
                        p.requires_grad_(False)
                    for p in model_mask_critic.parameters():
                        p.requires_grad_(False)
                    for p in model_imputer_critic.parameters():
                        p.requires_grad_(False)

                    data_noise.normal_()
                    fake_data = model_data_gen(data_noise)
                    
                    mask_noise.normal_()
                    fake_mask = model_mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask)
                    
                    impu_noise.uniform_()
                    imputed_data = model_imputer_gen(real_data, real_mask, impu_noise)
                    masked_imputed_data = mask_data(real_data, real_mask, imputed_data)

                    data_loss = -model_data_critic(masked_fake_data).mean()
                    mask_loss = -model_mask_critic(fake_mask).mean()
                    impu_loss = -model_imputer_critic(masked_imputed_data).mean()
                    
                    model_mask_gen.zero_grad()
                    (mask_loss + data_loss * alpha).backward(retain_graph=True)
                    mask_gen_optimizer.step()
                    
                    model_data_gen.zero_grad()
                    (data_loss + impu_loss * beta).backward(retain_graph=True)
                    data_gen_optimizer.step()

                    model_imputer_gen.zero_grad()
                    impu_loss.backward()
                    imputer_optimizer.step()

                    for p in model_data_critic.parameters():
                        p.requires_grad_(True)
                    for p in model_mask_critic.parameters():
                        p.requires_grad_(True)
                    for p in model_imputer_critic.parameters():
                        p.requires_grad_(True)

                if iter_trn % (max(n_iters//10000,1)) == 0:
                    run_trace2['TRN_ITER'].append(iter_trn)
                    run_trace2['TRN_EPOCH'].append(epoch_trn)
                    run_trace2['TRN_LOSS_G'].append(update_data_critic.loss_value)
                    run_trace2['TRN_LOSS_D'].append(update_impu_critic.loss_value)
                    #pbar.set_description('Training, epoch={}, loss g={:1.2e}'.format(
                    #    epoch_trn, loss_g.item()))
                    pbar.set_description('Training, Phase 2, epoch={}'.format(epoch_trn))
                pbar.update()
            
            # validation
            if epoch_trn % max(int(args.eval_freq*args.epoches),1) == 0:
                model_imputer_gen.eval()
                model_gnet = GNetWrapper(model_imputer_gen, impu_noise, mfv)
                # visual evaluation
                if not args.no_vis:
                    utils.eval_visual(test_loader, model_gnet, mfv, device, 
                            iter_trn, epoch_trn, args)
                # MSE/FID scores
                mse_score = utils.calc_mse_score(test_loader, mfv, model_g=model_gnet, 
                        dataset=args.dataset, device=device)
                rvalue_score = utils.calc_rvalue_score(test_loader, mfv, model_g=model_gnet, 
                        dataset=args.dataset, device=device)
                if args.dataset not in data.DENSE_DATASETS:
                    fid_score = utils.calc_fid_score(test_loader, mfv, model_g=model_gnet, 
                            dataset=args.dataset, device=device)
                else:
                    fid_score = np.nan

                # update trace
                run_trace2['VAL_ITER'].append(iter_trn)
                run_trace2['VAL_EPOCH'].append(epoch_trn)
                run_trace2['VAL_LOSS_G'].append(np.nan)
                run_trace2['VAL_LOSS_D'].append(np.nan)
                run_trace2['VAL_FID'].append(fid_score)
                run_trace2['VAL_MSE'].append(mse_score)
                run_trace2['VAL_R'].append(rvalue_score)
                utils.dump_trace(args.result_dir, run_trace2)

                # update the best model
                if best_dist >= fid_score and not np.isnan(fid_score):
                    best_dist = fid_score
                    best_model = copy.deepcopy(model_gnet).to(device='cpu:0')
                elif (best_dist >= mse_score and np.isnan(fid_score)) or args.missing_type == 'natural':
                    best_dist = mse_score
                    best_model = copy.deepcopy(model_gnet).to(device='cpu:0')
                else:
                    pass                    

                model_imputer_gen.train()

        pbar.close()
        self.model = best_model
        # preprate the model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        #torch.save(self, 
        #        args.data_dir + '/model/{}.pt'.format(args.conf_hash))




