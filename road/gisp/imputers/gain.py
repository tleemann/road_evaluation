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
import models.gsmv
import models.resnet
import data



class Imputer:
    def __init__(self):
        self.model = None
        self.mfv = None

    def __call__(self, x):
        return self.model(x)

    def to(self, x):
        self.model = self.model.to(x)
        return self
    
    def train(self, train_loader, test_loader, mfv, args):
        self.mfv = mfv
        assert(args.alpha > 0 and args.hint_rate > 0)
        run_trace = {}
        n_features = mfv.view(-1).size()[0]
        shape_orig = mfv.size()
        if len(args.device.split(',')) == 1:
            device = torch.device(args.device)
            if args.device[:4] == 'cuda':
                torch.cuda.set_device(int(args.device.split(':')[-1]))
        else:
            raise NotImplementedError
       
        # dnet output act
        if args.objective == 'hinge':
            act_out = None
        else:
            act_out = torch.nn.Sigmoid()

        # define models
        if args.dataset == 'mnist':
            model_gnet = models.gsmv.GNet_FC(n_features).to(device)
            model_dnetf = models.gsmv.DNetF_FC(n_features).to(device)
            #model_gnet = models.GNet_FCRes(n_features).to(device)
            #model_dnetf = models.DNetF_FCRes(n_features).to(device)
        elif args.dataset == 'cifar10' or args.dataset =='food-101':
            #model_gnet = models.GNet_UNet().to(device)
            #model_dnetf = models.DNetF_UNet().to(device)
            #model_gnet = models.gsmv.GNet_AttnNet(
            #        n_downsampling=1, n_blocks=4).to(device)
            #model_dnetf = models.gsmv.DNetF_AttnNet(
            #        n_downsampling=1, n_blocks=4, act_out=act_out).to(device) 
            model_gnet = models.gsmv.GNet_ResNet(
                    n_downsampling=1, n_blocks=4, attention=False).to(device)
            model_dnetf = models.gsmv.DNetF_ResNet(
                    n_downsampling=1, n_blocks=4, attention=False, act_out=act_out).to(device) 
        elif args.dataset == 'celeba':
            model_gnet = models.gsmv.GNet_ResNet().to(device)
            model_dnetf = models.gsmv.DNetF_ResNet().to(device)
        elif args.dataset in data.DENSE_DATASETS:
            model_gnet = models.gsmv.GNet_FC(n_features, args.layer_count, 
                    args.layer_factor).to(device)
            model_dnetf = models.gsmv.DNetF_FC(n_features, args.layer_count, 
                    args.layer_factor).to(device)
            #model_gnet = models.gsmv.GNet_FCRes(n_features, 3, n_features).to(device)
            #model_dnetf = models.gsmv.DNetF_FCRes(n_features, 3, n_features).to(device)
        else:
            raise NotImplementedError
        # initialize models
        model_gnet.apply(models.gsmv.weights_init_normal)
        model_dnetf.apply(models.gsmv.weights_init_normal)
        # define optimizers
        opt_gnet = torch.optim.Adam(model_gnet.parameters(), lr=args.lr_g, betas=(0.5,0.999))
        opt_dnet = torch.optim.Adam(model_dnetf.parameters(), lr=args.lr_d, betas=(0.5,0.999))
        opt_sch_gnet = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gnet, 
                mode='min', factor=0.2, patience=int(args.lr_patience/args.eval_freq), 
                verbose=True, threshold=1.0e-4, 
                threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
        opt_sch_dnet = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dnet, 
                mode='min', factor=0.2, patience=int(args.lr_patience/args.eval_freq), 
                verbose=True, threshold=1.0e-4, 
                threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

        
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
        pbar = tqdm(total=n_iters)
        for epoch_trn in range(args.epoches):
            print('loading dataset... (this might take a short time)')
            missing_process = data.MissingProcess(args)
            train_dset, test_dset = data.load_dataset(args, missing_process)
            train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, 
                    drop_last=True, shuffle=True, num_workers=4, pin_memory=False)
            test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, 
                    drop_last=True, shuffle=False, num_workers=4, pin_memory=False)
            # multiphase obj support
            if epoch_trn in phases_objective:
                args.objective = phases_objective[epoch_trn]
            for ind_batch, ((x,x_m),y) in enumerate(train_loader):
                iter_trn += 1
                # wait for at least 10gb memory
                # utils.wait_for_memory(10)

                # enable train mode for all
                model_gnet.train()
                model_dnetf.train()
                
                # get a batch
                x = x.to(device) - mfv
                x_m = x_m.to(device) - mfv
                mask = ~torch.isnan(x_m)

                ## train dnet d mode
                # forward pass with only dnetf in training.
                x_g = model_gnet(x_m).detach() # call generator.
                x_i = torch.where(mask.bool(), x_m, x_g) # impute.
                hint = utils.generate_hint(mask, args)
                y_f_pred = model_dnetf(torch.cat([x_i,hint],dim=1)) # call imputation predictor.
                y_f_target = mask.float()
                
                # calculate the loss
                if args.objective == 'hinge':
                    loss_d = utils.lossfunc_hinge_d_f(y_f_pred, y_f_target)
                else:
                    loss_d = utils.lossfunc_adv_d_f(y_f_pred, y_f_target)
                # backprop
                if iter_trn % args.skip_d == 0:
                    opt_dnet.zero_grad()
                    loss_d.backward()
                    opt_dnet.step()
                #if iter_trn == 1000:
                #    pdb.set_trace()
                ## train gnet
                # forward path
                x_g = model_gnet(x_m)
                x_i = torch.where(mask.bool(), x_m, x_g)
                hint = utils.generate_hint(mask, args)
                y_f_pred = model_dnetf(torch.cat([x_i,hint],dim=1))
                y_f_target = mask.float()
                if args.objective == 'hinge':
                    loss_g  = utils.lossfunc_hinge_g_f(y_f_pred, y_f_target)
                else:
                    loss_g  = utils.lossfunc_adv_g_f(y_f_pred, y_f_target)
                
                loss_g += args.alpha * torch.mean(
                        mask.float()*(x_i-x_g)**2 / (torch.mean(mask.float().view(-1,1), dim=0).reshape(-1,1,1,1)+1.0e-06)) # smoothed loss with the ground truth? 

                # backprop
                if iter_trn % args.skip_g == 0: 
                    opt_gnet.zero_grad()
                    loss_g.backward()
                    opt_gnet.step()
                
                if iter_trn % (max(n_iters//10000,1)) == 0:
                    run_trace['TRN_ITER'].append(iter_trn)
                    run_trace['TRN_EPOCH'].append(epoch_trn)
                    run_trace['TRN_LOSS_G'].append(loss_g.item())
                    run_trace['TRN_LOSS_D'].append(loss_d.item())
                    pbar.set_description('Training, epoch={}, loss g={:1.2e}, d={:1.2e}'.format(
                        epoch_trn, loss_g.item(), loss_d.item()))
                pbar.update()

                # if iter_trn == 3:
                #     break
            
            # validation
            if epoch_trn % max(int(args.eval_freq*args.epoches),1) == 0:
                # fixed batch for valid/test
                (x_val, x_m_val), y_val = next(iter(test_loader))
                x_val = x_val.to(device) - mfv
                x_m_val = x_m_val.to(device) - mfv
                mask_val = ~torch.isnan(x_m_val)
                with torch.no_grad():
                    # disable train mode
                    model_gnet.eval()
                    model_dnetf.eval()
                    # forward path
                    x_g_val = model_gnet(x_m_val).detach()
                    x_i_val = torch.where(mask_val.bool(), x_m_val, x_g_val)
                    hint_val = utils.generate_hint(mask_val, args)
                    y_f_pred_val = model_dnetf(torch.cat([x_i_val,hint_val],dim=1))
                    y_f_target_val = mask_val.float()
                    # calculate the loss
                    if args.objective == 'hinge':
                        loss_d_val = utils.lossfunc_hinge_d_f(y_f_pred_val, y_f_target_val)
                    else:
                        loss_d_val = utils.lossfunc_adv_d_f(y_f_pred_val, y_f_target_val)
                    ## validate gnet
                    # forward path
                    x_g_val = model_gnet(x_m_val)
                    x_i_val = torch.where(mask_val.bool(), x_m_val, x_g_val) # imputed x.
                    hint_val = utils.generate_hint(mask_val, args)
                    y_f_pred_val = model_dnetf(torch.cat([x_i_val,hint_val],dim=1))
                    y_f_target_val = mask_val.float()
                    
                    # calculate the loss
                    if args.objective == 'hinge':
                        loss_g_val  = utils.lossfunc_hinge_g_f(y_f_pred_val, y_f_target_val)
                    else:
                        loss_g_val  = utils.lossfunc_adv_g_f(y_f_pred_val, y_f_target_val)
                    loss_g_val += args.alpha * torch.mean(
                            mask_val.float()*(x_i_val-x_g_val)**2) / (torch.mean(mask_val.float())+1.0e-06)
                # visual evaluation
                if not args.no_vis:
                    utils.eval_visual(test_loader, model_gnet, mfv, device, 
                            iter_trn, epoch_trn, args, model_dnetf)
                # MSE/FID scores
                mse_score = utils.calc_mse_score(test_loader, mfv, model_g=model_gnet, 
                        dataset=args.dataset, device=device)

                # rvalue_score = utils.calc_rvalue_score(test_loader, mfv, model_g=model_gnet, 
                #         dataset=args.dataset, device=device)
                # print('rvalue: %.4f'%rvalue_score)
        
                rvalue_score = 0.0
                if args.dataset not in data.DENSE_DATASETS:
                    fid_score = utils.calc_fid_score(test_loader, mfv, model_g=model_gnet, 
                            dataset=args.dataset, device=device)
                else:
                    fid_score = np.nan

                # update trace
                run_trace['VAL_ITER'].append(iter_trn)
                run_trace['VAL_EPOCH'].append(epoch_trn)
                run_trace['VAL_LOSS_G'].append(loss_g_val.item())
                run_trace['VAL_LOSS_D'].append(loss_d_val.item())
                run_trace['VAL_FID'].append(fid_score)
                run_trace['VAL_MSE'].append(mse_score)
                run_trace['VAL_R'].append(rvalue_score)
                utils.dump_trace(args.result_dir, run_trace)

                # update the best model
                if best_dist >= fid_score and not np.isnan(fid_score):
                    best_dist = fid_score
                    best_model = copy.deepcopy(model_gnet).to(device='cpu:0')
                elif (best_dist >= mse_score and np.isnan(fid_score)) or args.missing_type == 'natural':
                    best_dist = mse_score
                    best_model = copy.deepcopy(model_gnet).to(device='cpu:0')
                else:
                    pass
                # lr decay schedulder
                opt_sch_gnet.step(best_dist)
                opt_sch_dnet.step(best_dist)
                for param in best_model.parameters():
                    param.requires_grad = False
                curr_dict = {}
                curr_dict["generator"] = best_model.state_dict()
                curr_dict["mfv"] = mfv
                torch.save(curr_dict, args.result_dir + '/model_{}.pt'.format(args.conf_hash))
                print('model saved for epoch: %d'%epoch_trn)
        pbar.close()
        self.model = best_model
        # preprate the model
        self.model.eval()

        torch.save(self, args.data_dir + '/model/{}.pt'.format(args.conf_hash))
        



