
import time
import pdb
import sys
import os
import random
import imghdr
from PIL import Image
import pickle

import numpy as np
import scipy.stats
from scipy import linalg
from scipy import stats
import matplotlib.pyplot as plt
import torch
import torchvision
import psutil
import sklearn.metrics

import data

sys.path.append("..")

from otherwork.pytorch_fid import inception as pytfid_inception
from otherwork.pytorch_fid import lenet as pytfid_lenet

def get_activation(act_name):
    # activation function
    if act_name == 'relu':
        act_fn = torch.nn.ReLU
    elif act_name == 'tanh':
        act_fn = torch.nn.Tanh
    elif act_name == 'sigmoid':
        act_fn = torch.nn.Sigmoid
    elif act_name == 'selu':
        act_fn = torch.nn.SELU
    else:
        raise NotImplementedError(act_name)
    return act_fn

def wait_for_memory(free_gb=10):
    t = 0
    flag_waited = False
    while psutil.virtual_memory().available / 1e9 < free_gb:
        print('Waiting for memory: free {:.1f}/{}, time:{}m'.format(
            psutil.virtual_memory().available/1e9, free_gb, t), end='\r')
        time.sleep(60)
        t += 1
        flag_waited = True
    if flag_waited:
        print('Waiting for memory: Done')

def lossfunc_adv_d_f(y, mask):
    mask = mask.float() #* 0.9
    return -torch.mean(mask * torch.log(y+1.0e-8) + (1.0-mask) * torch.log(1.0-y+1.0e-8))

def lossfunc_hinge_d_f(y, mask):
    mask = mask.float() #* 0.9
    #y = torch.log(1.0e-5 + y/(1.0e-5+1.0-y))
    return -torch.mean(mask * torch.nn.ReLU()(-1+y) + (1.0-mask) * torch.nn.ReLU()(-1.0-y))

def lossfunc_adv_g_f(y, mask):
    mask = mask.float() #* 0.9
    return -torch.mean((1.0-mask) * torch.log(y+1.0e-8)) / torch.mean(1.0-mask)

def lossfunc_hinge_g_f(y, mask):
    mask = mask.float() #* 0.9
    #y = torch.log(1.0e-5 + y/(1.0e-5+1.0-y))
    return -torch.mean((1.0-mask) * y) / torch.mean(1.0-mask)

def lossfunc_mse_g(x_m, x_g):
    mask = 1 - torch.isnan(x_m)
    x_m = torch.where(mask, x_m, x_g)
    return torch.mean((x_g-x_m)**2)

def lossfunc_ce(y_pred, y_target):
    y_pred = y_pred.view((-1,))
    y_target = y_target.view((-1,))
    y_target = y_target.float() #* 0.9
    return -torch.mean((1.0-y_target) * torch.log(1.0-y_pred+1.0e-8) + \
            y_target * torch.log(y_pred+1.0e-8))
    #return torch.nn.BCELoss()(y_pred, y_target)

def generate_hint(mask, args):
    mask = mask.float()
    if len(mask.shape) > 2:
        mask = torch.unsqueeze(mask[:,0], 1)
    b = torch.bernoulli(torch.ones_like(mask)*args.hint_rate)
    hint = b * mask + 0.5 * (1.0 - b)
    return hint

def dump_trace(path, trace):
    with open(path+'trace.pkl', 'wb+') as f:
        pickle.dump(trace, f)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(trace['TRN_ITER'], trace['TRN_LOSS_G'])
    plt.plot(trace['VAL_ITER'], trace['VAL_LOSS_G'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.title('GNet')

    plt.subplot(3,1,2)
    plt.plot(trace['TRN_ITER'], trace['TRN_LOSS_D'])
    plt.plot(trace['VAL_ITER'], trace['VAL_LOSS_D'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.title('DNet')
    
    ax1 = plt.subplot(3,1,3)
    ax1.plot(trace['VAL_EPOCH'], trace['VAL_FID'], 'r')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID', color='r')
    ax2 = ax1.twinx()
    ax2.plot(trace['VAL_EPOCH'], trace['VAL_MSE'], 'b')
    ax2.set_ylabel('MSE', color='b')
    ax1.set_title('Scores')

    plt.tight_layout()
    plt.savefig(path+'learning_curve.png')
    plt.close('all')


def dump_trace_p(path, trace):
    with open(path+'trace_p.pkl', 'wb+') as f:
        pickle.dump(trace, f)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(trace['TRN_ITER'], trace['TRN_LOSS'])
    plt.plot(trace['VAL_ITER'], trace['VAL_LOSS'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.title('PNet')

    plt.subplot(2,1,2)
    plt.plot(trace['VAL_EPOCH'], trace['VAL_ACCURACY'], 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (best:{:.2f}%)'.format(100*np.max(trace['VAL_ACCURACY'])))

    plt.tight_layout()
    plt.savefig(path+'learning_curve_p.png')
    plt.close('all')


def calc_inception_score(dataloader, model_g=None, device='cpu:0', resize=True, splits=1):
    n_batches = len(dataloader)
    if model_g is not None:
        model_g = model_g.to(device)
    # Load inception model
    inception_model = torchvision.models.inception.inception_v3(pretrained=True, 
            transform_input=False).to(device)
    inception_model.eval()
    def get_pred(x):
        if resize:
            x = torch.nn.functional.interpolate(x, size=(299,299), 
                    mode='bilinear', align_corners=False)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1)

    # Get predictions
    preds = []
    for i, ((x,x_m),y) in enumerate(dataloader, 0):
        # wait for at least 10gb memory
        wait_for_memory(10)
        with torch.no_grad():
            if model_g is not None:
                x_m = x_m.to(device)
                x_g = model_g(x_m)
                x_i = torch.where(torch.isnan(x_m), x_g, x_m)
            else:
                x_i = x.to(device)
            # store on cpu mem
            pred = get_pred(x_i).to('cpu:0').numpy()
            preds.append(pred)
    preds = np.vstack(preds)
    N = preds.shape[0]
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(scipy.stats.entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calc_mse_score(dataloader, mfv, model_g, dataset, device='cpu:0', resize=True):
    n_batches = len(dataloader)
    mfv = mfv.to(device)
    if model_g is not None:
        model_g = model_g.to(device)
    # Get predictions
    mse = 0.0
    for i, ((x,x_m),y) in enumerate(dataloader, 0):
        # wait for at least 10gb memory
        wait_for_memory(10)
        with torch.no_grad():
            x_m = x_m.to(device) - mfv
            x_g = model_g(x_m)
            x_i = torch.where(torch.isnan(x_m), x_g, x_m) + mfv
            x = x.to(device)
            # calc mse
            mse += torch.mean((x-x_i)**2.0).item()
    mse /= n_batches
    return mse


def calc_rvalue_score(dataloader, mfv, model_g, dataset, device='cpu:0', resize=True):
    n_batches = len(dataloader)
    mfv = mfv.to(device)
    if model_g is not None:
        model_g = model_g.to(device)
    # Get predictions
    pred = []
    target = []
    for i, ((x,x_m),y) in enumerate(dataloader, 0):
        # wait for at least 10gb memory
        wait_for_memory(10)
        with torch.no_grad():
            x_m = x_m.to(device) - mfv
            x_g = model_g(x_m)
            x_i = torch.where(torch.isnan(x_m), x_g, x_m) + mfv
            x = x.to(device)
            # calc mse
            pred.append(x_i.detach().cpu().numpy().ravel())
            target.append(x.detach().cpu().numpy().ravel())
    pred = np.array(pred).ravel()
    target = np.array(target).ravel()
    slope, intercept, r_value, p_value, std_err = stats.linregress(pred,target)
    return r_value

def calc_fid_score(dataloader, mfv, model_g, dataset, device='cpu:0', resize=True):
    eps=1e-6
    n_batches = len(dataloader)
    mfv = mfv.to(device)
    if model_g is not None:
        model_g = model_g.to(device)
    # Load inception model
    if dataset == 'mnist':
        model_incept = pytfid_lenet.LeNet().to(device)
    else:
        model_incept = pytfid_inception.InceptionV3(resize_input=False).to(device)
    model_incept.eval();

    def get_act(x):
        if resize and dataset !='mnist':
            x = torch.nn.functional.interpolate(x, size=(299,299), 
                    mode='bilinear', align_corners=False)
        x = model_incept(x)
        return x

    # Get predictions
    acts_x = []
    acts_g = []
    for i, ((x,x_m),y) in enumerate(dataloader, 0):
        # wait for at least 10gb memory
        wait_for_memory(10)
        with torch.no_grad():
            x_m = x_m.to(device) - mfv
            x_g = model_g(x_m)
            x_i = torch.where(torch.isnan(x_m), x_g, x_m) + mfv
            x = x.to(device)
            # store on cpu mem
            act_g = get_act(x_i)[0].squeeze().to('cpu:0').numpy()
            act_x = get_act(x)[0].squeeze().to('cpu:0').numpy()
            acts_g.append(act_g)
            acts_x.append(act_x)
    acts_g = np.vstack(acts_g)
    acts_x = np.vstack(acts_x)

    # calc stats
    mu_g = np.mean(acts_g, axis=0)
    mu_x = np.mean(acts_x, axis=0)
    sigma_g = np.cov(acts_g, rowvar=False)
    sigma_x = np.cov(acts_x, rowvar=False)
    
    # calc fid score
    diff = mu_g - mu_x
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_x.dot(sigma_g), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_x + offset).dot(sigma_g + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma_x) + np.trace(sigma_g) - 2 * tr_covmean)


def eval_visual(test_loader, model_gnet, mfv, device, 
        iter_trn, epoch_trn, args, model_dnetf=None):
    # make dirs
    out_dir = args.result_dir + '/vis/'
    os.makedirs(out_dir, exist_ok=True)
    # move to device
    model_gnet = model_gnet.to(device)
    mfv = mfv.to(device)
    # get a batch
    (x_val, x_m_val), y_val = next(iter(test_loader))
    x_val = x_val.to(device) - mfv
    x_m_val = x_m_val.to(device) - mfv
    x_g_val = model_gnet(x_m_val)
    mask_val = ~torch.isnan(x_m_val)
    x_i_val = torch.where(mask_val.bool(), x_m_val, x_g_val)
    with torch.no_grad():
        # prepare image output
        imgs_g = [x_val.to('cpu:0'), x_m_val.to('cpu:0')] # Xval, xval masked
        imgs_i = [x_val.to('cpu:0'), x_m_val.to('cpu:0')]
        if model_dnetf:
            model_dnetf = model_dnetf.to(device)
            hint_val = generate_hint(mask_val, args)
            y_f_pred_val = model_dnetf(torch.cat([x_i_val,hint_val],dim=1))
            imgs_g.append((y_f_pred_val-mfv).to('cpu:0'))
            imgs_i.append((y_f_pred_val-mfv).to('cpu:0'))

        if args.dataset not in data.DENSE_DATASETS:
            N_SAMPLES = 16
        else:
            N_SAMPLES = 128
        for _ in range(N_SAMPLES):
            x_m_val = x_m_val.to(device)
            x_g_val = model_gnet(x_m_val)
            x_i_val = torch.where(mask_val.bool(), x_m_val, x_g_val)
            imgs_g.append(x_g_val.to('cpu:0'))
            imgs_i.append(x_i_val.to('cpu:0'))
        # save output files
        if args.dataset not in data.DENSE_DATASETS:
            imgs_g = torch.clamp(torch.cat(imgs_g) + mfv.to('cpu:0'), 0, 1)
            imgs_i = torch.clamp(torch.cat(imgs_i) + mfv.to('cpu:0'), 0, 1)
            imgs_g = torch.where(torch.isnan(imgs_g), torch.tensor(0.0), imgs_g)
            imgs_i = torch.where(torch.isnan(imgs_i), torch.tensor(0.0), imgs_i)
            torchvision.utils.save_image(imgs_g, 
                        out_dir+'images_g_obj{}_epoch{}_iter{}.png'.format(
                            args.objective,epoch_trn, iter_trn), 
                        nrow=x_m_val.shape[0], pad_value=0.5)
            torchvision.utils.save_image(imgs_i, 
                        out_dir+'images_i_obj{}_epoch{}_iter{}.png'.format(
                            args.objective, epoch_trn,iter_trn), 
                        nrow=x_m_val.shape[0], pad_value=0.5)
        else:
            n_features = x_val.shape[1]
            imgs_g = torch.cat([(a+mfv.to('cpu:0')).view(1,-1) for a in imgs_g], 0).numpy()
            imgs_i = torch.cat([(a+mfv.to('cpu:0')).view(1,-1) for a in imgs_i], 0).numpy()
            imgs_g = np.insert(imgs_g, 3, np.nan, axis=0)
            imgs_i = np.insert(imgs_i, 3, np.nan, axis=0)
            inds = np.arange(n_features, imgs_g.shape[1], n_features)
            imgs_g = np.insert(imgs_g, inds, np.nan, axis=1)
            imgs_i = np.insert(imgs_i, inds, np.nan, axis=1)
            imgs_g = imgs_g[:,:1000]
            imgs_i = imgs_i[:,:1000]
            plt.matshow(imgs_g, vmin=0.0, vmax=1.0, cmap='jet')
            plt.axis('off')
            plt.savefig(out_dir+'images_g_obj{}_epoch{}_iter{}.png'.format(
                           args.objective, epoch_trn,iter_trn), 
                           interpolation='none', dpi=600) 
            plt.matshow(imgs_i, vmin=0.0, vmax=1.0, cmap='jet')
            plt.axis('off')
            plt.savefig(out_dir+'images_i_obj{}_epoch{}_iter{}.png'.format(
                           args.objective, epoch_trn,iter_trn), 
                           interpolation='none', dpi=600)
            plt.close('all')
            np.save(out_dir+'images_i_obj{}_epoch{}_iter{}.npy'.format(
                args.objective,epoch_trn,iter_trn), imgs_i)


def eval_ensemble(test_loader, model_gnet, model_pnet, mfv, 
        device, iter_trn, epoch_trn, args, plot=True):
    N_SAMPLES = args.n_samples #64
    # make output dirs
    out_dir = args.result_dir + 'vis_p/'
    os.makedirs(out_dir, exist_ok=True)
    # move to device
    model_gnet = model_gnet.to(device)
    model_pnet = model_pnet.to(device)
    mfv = mfv.to(device)
    # get batches    
    ens_conf = []
    ens_acc = []
    ys_label = []
    ys_prob = []
    for (x, x_m), y in test_loader:
        # wait for at least 10gb memory
        wait_for_memory(10)
        #x = x.to(device) - mfv
        x_m = x_m.to(device) - mfv
        mask = 1 - torch.isnan(x_m)
        ys_label.append(y.cpu())
        with torch.no_grad():
            btc_preds = []
            btc_probs = []
            btc_xs = [(x_m+mfv).cpu().numpy()]
            for _ in range(N_SAMPLES):
                x_g = model_gnet(x_m)
                x_i = torch.where(mask.bool(), x_m, x_g)
                y_out = model_pnet(x_i)
                _, y_pred = torch.max(y_out, 1)
                y_pred = y_pred.view(-1, 1)
                btc_preds.append(y_pred)
                btc_probs.append(torch.nn.functional.softmax(y_out,-1).cpu().numpy())
                btc_xs.append((x_i+mfv).cpu().numpy())
        btc_preds = torch.cat(btc_preds, dim=1).cpu().numpy()
        btc_mode, btc_cnt = scipy.stats.mode(btc_preds, 1)
        btc_conf = btc_cnt.astype(np.float) / btc_preds.shape[1]
        btc_acc = (btc_mode.ravel()==y.cpu().numpy()).astype(np.float)
        ens_conf.append(btc_conf.ravel())
        ens_acc.append(btc_acc)
        btc_probs = np.stack(btc_probs).mean(0)
        ys_prob.append(btc_probs)

    ens_accuracy = np.mean(ens_acc)
    # acc/conf
    ens_conf = np.hstack(ens_conf)
    ens_acc = np.hstack(ens_acc)
    bins = np.linspace(0.0, 1.0, 10)
    conf_digits = np.digitize(ens_conf, bins)
    accu_at_bins = []
    for ind_bin in range(len(bins)):
        if np.any(conf_digits==ind_bin):
            accu_at_bins.append(np.mean(ens_acc[conf_digits==ind_bin]))
        else:
            accu_at_bins.append(np.nan)
    
    # AUROC
    ys_prob = np.vstack(ys_prob)
    ys_label = torch.cat(ys_label).view(-1,1)
    ys_label_1h = torch.zeros(ys_label.shape[0], 1+ys_label.max())
    ys_label_1h.scatter_(1,ys_label,1)
    try:
        auroc = sklearn.metrics.roc_auc_score(ys_label_1h.numpy(), ys_prob)
    except ValueError:
        auroc = float('nan')

    # if plot is enabled
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(bins, accu_at_bins)
        plt.plot(bins, bins)
        plt.xlabel('Certainty')
        plt.ylabel('Accuracy')
        plt.title('Ensemble Accuracy: {}%'.format(100*ens_accuracy))
        plt.subplot(2,1,2)
        plt.hist(ens_conf)
        plt.savefig(out_dir+'conf_i_obj{}_epoch{}_iter{}.png'.format(
                        args.objective, epoch_trn,iter_trn))

    if args.dump_ens:
        with open(out_dir+'ens_vis_i_obj{}_epoch{}_iter{}.pkl'.format(
            args.objective, epoch_trn,iter_trn), 'wb+') as f:
            pickle.dump({'btc_xs':btc_xs[:9],'btc_probs':btc_probs[:8,:], 
                'btc_preds':btc_probs},f)



    return {'ENS_ACCURACY': ens_accuracy, 'ENS_BINS': bins, 'ENS_ACCUS': accu_at_bins, 
            'ENS_AUROC': auroc}



