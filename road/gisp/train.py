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
import models.gsmv, models.resnet, models.mlp
import data
import imputers.autoencoder, imputers.gsmv, imputers.gain, imputers.mean, imputers.mice
#imputers.misgan, 

parser = argparse.ArgumentParser()
# path setup
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--noise_std', type=float, default=0.0)
parser.add_argument('--data_dir', type=str, default='~/Database/Image/')
parser.add_argument('--result_dir', type=str, default='./run_outputs/')
# training setup
parser.add_argument('--epoches', type=int, default=100000)
parser.add_argument('--eval_freq', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cpu:0')
# missingness setup
parser.add_argument('--missing_type', type=str, 
        choices=['mcar_uniform', 'mcar_rect', 'mcar_rectinv', 'natural',
                'mnar_foreground', 'mnar_background'], 
        default='mcar_uniform')
parser.add_argument('--missing_rate', type=float, default=0.5)
# network setup
parser.add_argument('--objective', type=str, 
        choices=['bce', 'hinge', 'autoencoder', 'misgan', 'gain', 'mean', 'mice'])
parser.add_argument('--lr_g', type=float, default=0.0005)
parser.add_argument('--lr_d', type=float, default=0.0005)
parser.add_argument('--skip_g', type=int, default=1)
parser.add_argument('--skip_d', type=int, default=1)
parser.add_argument('--lr_patience', type=float, default=0.25)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mfv_samples', type=int, default=10000)
parser.add_argument('--hint_rate', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--n_samples', type=int, default=64)
# predictor setup
parser.add_argument('--use_cache', action='store_true')
parser.add_argument('--no_vis', action='store_true')
parser.add_argument('--dump_ens', action='store_true')
parser.add_argument('--train_predictor', action='store_true')
parser.add_argument('--layer_count', type=int, default=3)
parser.add_argument('--layer_factor', type=float, default=1.0)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--w_decay_log', type=float, default=-20.0)
parser.add_argument('--activation', type=str, default='sigmoid')
parser.add_argument('--aug_noise_std', type=float, default=0.0)

torch.set_num_threads(2)

def initialize_exp(args):
    start_time = time.strftime("%m_%d_%Y__%H_%M_%S", time.localtime())
    args_dict = vars(args)
    args.conf_hash = hash(str(
        [str(args_dict[a]) for a in args_dict if a not in ['device']])) % 2**32
    args.exp = args.dataset + '_' + args.objective + '_' \
            + args.exp + '_{}{}_pid{}'.format(
                    args.missing_type, args.missing_rate, os.getpid())
    args.data_dir = os.path.expanduser(args.data_dir) + args.dataset + '/'
    args.result_dir = os.path.expanduser(args.result_dir) + args.exp + '/'
    print('Time:', start_time, 'Exp Name:', args.exp)
    print(args)
    if len(args.device.split(',')) == 1:
        device = torch.device(args.device)
        if args.device[:4] == 'cuda':
            torch.cuda.set_device(int(args.device.split(':')[-1]))
    else:
        raise NotImplementedError
    # create result/model directories (if necessary)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.data_dir + '/model/', exist_ok=True)
    # save training config
    with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
        json.dump(args_dict, f)

    # make the missing generator 
    #missing_process = data.MissingProcess(mtype='mcar_uniform', args={'missing_rate':0.50})
    missing_process = data.MissingProcess(args)
    # dataset
    print('loading dataset... (it may take a few minutes)')
    train_dset, test_dset = data.load_dataset(args, missing_process)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, 
            drop_last=True, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size//2, 
            drop_last=True, shuffle=False, num_workers=4, pin_memory=False)

    # compute the mean pixel value of train dataset
    mean_fv = 0.
    inds_mean = np.random.choice(len(train_dset),
            size=min(args.mfv_samples, len(train_dset)), replace=False)
    #inds_mean = np.arange(min(args.mfv_samples, len(train_dset)))
    pbar = tqdm(total=len(inds_mean), desc='Computing the mean feature values')
    for ind in inds_mean:
        (x, x_m), label = train_dset.__getitem__(ind)
        mean_fv += x.detach()
        pbar.update()
    mean_fv /= len(inds_mean)
    pbar.close()
    mfv = mean_fv.detach().to(device)
    return train_loader, test_loader, mfv, args


def train_predictor(train_loader, test_loader, mfv, model_gnet, args):
    args.epoches = min(1000, args.epoches)
    run_trace = {}
    n_features = mfv.view(-1).size()[0]
    n_classes = data.DATASET_CLASSES[args.dataset]
    shape_orig = mfv.size()
    if len(args.device.split(',')) == 1:
        device = torch.device(args.device)
        if args.device[:4] == 'cuda':
            torch.cuda.set_device(int(args.device.split(':')[-1]))
    else:
        raise NotImplementedError
   
    mfv = mfv.to(device)
    # define models
    model_gnet = model_gnet.to(device)

    if args.dataset == 'mnist':
        raise NotImplementedError
    elif args.dataset == 'cifar10' or args.dataset == 'food-101':
        model_pnet = models.resnet.ResNet18(num_classes=n_classes)
    elif args.dataset in data.DENSE_DATASETS:
        #model_pnet = models.mlp.PNet_FC(n_features, n_features//2, n_classes)
        model_pnet = models.mlp.MLP(n_in=n_features, n_out=n_classes, 
                n_layers=args.layer_count, layer_factor=args.layer_factor, 
                act_fn=args.activation, dropout_p=args.dropout_p)
    else:
        raise NotImplementedError
    model_pnet = model_pnet.to(device)
    # define optimizers
    opt_pnet = torch.optim.Adam(model_pnet.parameters(), 
            lr=0.001, betas=(0.9,0.999), weight_decay=10**args.w_decay_log) 
    opt_sch_pnet = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pnet, 
            mode='max', factor=0.2, patience=int(args.lr_patience/args.eval_freq), 
            verbose=True, threshold=0.01, 
            threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-8)

    # loss fucn
    lossfunc_p = torch.nn.CrossEntropyLoss()

    # train
    run_trace['TRN_ITER'] = []
    run_trace['TRN_EPOCH'] = []
    run_trace['TRN_LOSS'] = []
    run_trace['VAL_ITER'] = []
    run_trace['VAL_EPOCH'] = []
    run_trace['VAL_LOSS'] = []
    run_trace['VAL_ACCURACY'] = [] 
    run_trace['VAL_AUROC'] = [] 
    run_trace['VAL_ENS'] = []
    best_model = None
    best_score = -np.inf
    iter_trn = 0
    n_batches = len(train_loader)
    n_iters = args.epoches * n_batches
    pbar = tqdm(total=n_iters)
    for epoch_trn in range(args.epoches):
        for ind_batch, ((x,x_m),y) in enumerate(train_loader):
            iter_trn += 1
            # wait for at least 10GB memory
            utils.wait_for_memory(10)

            # enable train mode for all
            model_pnet.train()
            
            # get a batch
            #x = x.to(device) - mfv
            y = y.view(-1).to(device)
            x_m = x_m.to(device) - mfv
            mask = 1 - torch.isnan(x_m)

            ## train pnet
            # forward path
            x_g = model_gnet(x_m).detach()
            x_i = torch.where(mask.bool(), x_m, x_g)
            # resize x_i
            #x_i_resize = torch.nn.functional.interpolate(x_i, 
            #        size=(224,224), mode='bilinear', align_corners=False)
            # add Guassian noise to input
            if args.aug_noise_std != 0.0:
                x_i += args.aug_noise_std * torch.randn_like(x_i)
            y_out = model_pnet(x_i)

            # calculate the loss
            loss_p = lossfunc_p(y_out, y)
            # backprop
            if True:
                opt_pnet.zero_grad()
                loss_p.backward()
                opt_pnet.step()

            
            if iter_trn % (max(n_iters//10000,1)) == 0:
                run_trace['TRN_ITER'].append(iter_trn)
                run_trace['TRN_EPOCH'].append(epoch_trn)
                run_trace['TRN_LOSS'].append(loss_p.item())
                pbar.set_description('Training, epoch={}, loss ={:1.2e}'.format(
                    epoch_trn, loss_p.item()))
            pbar.update()
        
        # validation
        if epoch_trn % max(int(args.eval_freq*args.epoches),1) == 0:
            # disable train mode
            model_pnet.eval()
            loss_p_val = 0.0
            total = 0
            correct = 0
            for  ind_batch_val, ((x_val, x_m_val), y_val) in enumerate(test_loader):
                #x_val = x_val.to(device) - mfv
                y_val = y_val.view(-1).to(device)
                x_m_val = x_m_val.to(device) - mfv
                mask_val = 1 - torch.isnan(x_m_val)
                with torch.no_grad():
                    # forward path
                    x_g_val = model_gnet(x_m_val).detach()
                    x_i_val = torch.where(mask_val.bool(), x_m_val, x_g_val)
                    
                    # resize x_i
                    #x_i_val_resize = torch.nn.functional.interpolate(x_i_val, 
                    #        size=(224,224), mode='bilinear', align_corners=False)
                    y_out_val = model_pnet(x_i_val)
                    _, y_pred_val = torch.max(y_out_val, 1)
                    y_pred_val = y_pred_val.view(-1)
                    # calculate the loss
                    loss_p_val += lossfunc_p(y_out_val, y_val).item() / len(test_loader)
                    correct += torch.sum((y_pred_val==y_val).float()).item()
                    total += y_pred_val.shape[0]
            
            acc_val = correct / total
            # ensemble analysis
            ens_val = utils.eval_ensemble(test_loader, model_gnet, model_pnet, 
                    mfv, device, iter_trn, epoch_trn, args, plot=(not args.no_vis))
            #if args.objective in ['bce', 'hinge']:
            acc_val = ens_val['ENS_ACCURACY']
            # save output files
            # update the best model
            if best_score <= acc_val:
                best_score = acc_val
                best_model = copy.deepcopy(model_pnet).to(device='cpu:0')
                # save the model
                #torch.save(best_model, 
                #        args.data_dir + '/model/{}.pt'.format(args.conf_hash))
                #run_trace.update(ens_val)
            # lr decay schedulder
            opt_sch_pnet.step(best_score)
            # update trace
            run_trace['VAL_ITER'].append(iter_trn)
            run_trace['VAL_EPOCH'].append(epoch_trn)
            run_trace['VAL_LOSS'].append(loss_p_val)
            run_trace['VAL_ACCURACY'].append(acc_val)
            run_trace['VAL_AUROC'].append(ens_val['ENS_AUROC'])
            run_trace['VAL_ENS'].append(ens_val)
            utils.dump_trace_p(args.result_dir, run_trace)
                
    pbar.close()
    return best_model

if __name__ == '__main__':
    print('')
    print(79*'#')
    
    print('Initializing the experiment...')
    args = parser.parse_args()
    train_loader, test_loader, mfv, args = initialize_exp(args)
    print('Initializing the experiment... DONE')
    print(39*'#')
    
    model_path = args.data_dir + '/model/{}.pt'.format(args.conf_hash)

    if args.use_cache and os.path.exists(model_path):
        print('Loading the cached model...')
        imputer = imputers.base.Imputer()
        imputer.load(model_path)
        print('Loading the cached model... DONE')
    else:
        print('Training the imputer...')
        if args.objective == 'autoencoder':
            imputer = imputers.autoencoder.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        elif args.objective == 'misgan':
            imputer = imputers.misgan.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        elif args.objective == 'gain':
            imputer = imputers.gain.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        elif args.objective == 'mean':
            imputer = imputers.mean.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        elif args.objective == 'mice':
            imputer = imputers.mice.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        else:
            imputer = imputers.gsmv.Imputer()
            imputer.train(train_loader, test_loader, mfv, args)
        print('Training the imputer... DONE')
    print(39*'#')
    
    if args.train_predictor:
        print('Training the predictor...')
        train_predictor(train_loader, test_loader, 
                mfv, imputer, args)
        print('Training the predictor... DONE')


    print(79*'#')

