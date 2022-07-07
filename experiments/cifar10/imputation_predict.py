# Repeatedly train the imputation predictor to assess mask leakage in different attribution maps.
# See comments in ImputationPrediction.py for some more descriptions.

from logging import PercentStyle
from unittest.util import three_way_cmp
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image

import json

from road.utils import *
from road.gpu_dataloader import ImputedDatasetMasksOnly
from road.imputations import ZeroImputer, ChannelMeanImputer, NoisyLinearImputer
device="cuda:1"
batch_size = 32
learning_rate = 5e-5


def create_data_sets(p=0.5, use_roar=True):
    """ Return a test and train loader with randomly blacked out samples that are either mean-value imputed or linearly imputed."""
    transform_tensor = transforms.ToTensor()
    explanation_train, explanation_test, prediction_train, prediction_test  = load_expl('../../data/ig/base_train.pkl', '../../data/ig/base_test.pkl')
    explanation_test = np.random.randn(*np.stack(explanation_test).shape)
    explanation_train = np.random.randn(*np.stack(explanation_train).shape)

    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_tensor)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_tensor)

    trainset = ImputedDatasetMasksOnly(cifar_train, mask=explanation_train, th_p=p, remove=use_roar)

    testset = ImputedDatasetMasksOnly(cifar_test, mask=explanation_test, th_p=p, remove=use_roar)

    return trainset, testset


class ImputationPredictor(nn.Module):
    def __init__(self, th_p = 0.5):
        super(ImputationPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 128, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.conv4.bias.data = torch.tensor(np.log((1.0-th_p)/th_p),dtype=torch.float).reshape(1) # initialize bias to match the expected value.
        
    def forward(self, x):
        x1 = torch.sigmoid(self.conv1(x))
        #x = torch.relu(self.conv1b(x))
        x2 = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x1+x2))
        x = torch.sigmoid(self.conv4(x))
        return x


def test_imp_predictor(model, testloader, use_imputation):
    """ """
    model.eval()
    model.to(device)
    cum_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, _, _, masks = data
            inputs = inputs.to(device)
            masks = masks.to(device)
            real_outputs = masks.to(device)
            outputs = model(use_imputation.batched_call(inputs, masks)-0.5).squeeze(1)
            #print(real_outputs.shape, outputs.shape)
            loss = -torch.sum((real_outputs*torch.log(outputs+1e-8) + (1-real_outputs)*torch.log(1-outputs+1e-8)))
            predicted = (outputs > 0.5)
            correct += (predicted == real_outputs).sum().item()
            total += np.prod(predicted.shape)
            cum_loss += loss.detach().item()
    #print(cum_loss, correct/total)    
    return correct/total


def train_imp_predictor(model, optim, trainloader, use_imputation):
    """
        trainexpl: Binary explanations.
    """
    ## eval the model
    model.eval()
    model.to(device)
    cum_loss = 0.0
    correct, total = 0, 0
    for data in tqdm(trainloader):
        inputs, _, _, masks = data
        inputs = inputs.to(device)
        masks = masks.to(device)

        real_outputs = masks.to(device)
        
        outputs = model(use_imputation.batched_call(inputs, masks)-0.5).squeeze(1)

        #print(real_outputs.shape, outputs.shape)
        loss1 = -torch.sum(real_outputs*torch.log(outputs+1e-8))
        loss2 = -torch.sum((1-real_outputs)*torch.log(1-outputs+1e-8))
        loss = loss2+loss1
        loss.backward()
        optim.step()
        predicted = (outputs > 0.5).float()
        #print(predicted.shape, masks.shape)
        #print(predicted.mean(), masks.float().mean())
        correct += (predicted == real_outputs).sum().item()
        total += np.prod(predicted.shape)
        #print(correct, total)
        cum_loss += loss.detach().item()
    ##### calculate the average probability
    print(cum_loss, (correct/total)*100, "% acc.")    
    return model

if __name__ == "__main__":
    """ Main functions. Test different percentages, fixed/linear imputations and perform multiple repetitions.
        Save the results to some JSON file named imputation_results.json in a dict format.
    """
    percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    imputations = ["fixed", "linear"]
    my_imputers = {"fixed": ZeroImputer(), "linear": NoisyLinearImputer(noise=0.05)}

    if os.path.exists("imputation_results.json"):
        results = json.load(open("imputation_results.json"))
        print("Loading old results...")
    else:
        results = {}

        results["thresholds"] = percentages
        results["imputations"] = imputations
        results["data"] = {}
    for reps in range(10):
        for th_p in percentages:
            if str(th_p) not in results["data"]:
                results["data"][str(th_p)] = {}
            for imputation in ["fixed", "linear"]:
                print("Imputation:", imputation, "Percentage:", th_p)
                train_set_org, test_set_org = create_data_sets(p=th_p)
                trainloader = torch.utils.data.DataLoader(train_set_org, batch_size=batch_size, shuffle=True, num_workers=8)
                testloader = torch.utils.data.DataLoader(test_set_org, batch_size=batch_size, shuffle=False, num_workers=8)
                imppred = ImputationPredictor(th_p)
                myopt = optim.Adam(imppred.parameters(), lr = learning_rate)
                acc = 0.0
                last_acc = 0.0
                ep_no_improve = 0 # epoch w/o improvement
                for epoch in range(100):
                    imppred = train_imp_predictor(imppred, myopt, trainloader, my_imputers[imputation])
                    acc = test_imp_predictor(imppred, testloader, my_imputers[imputation])
                    if acc < last_acc:
                        ep_no_improve += 1
                        if ep_no_improve==2:
                            break
                    else:
                        ep_no_improve = 0
                    last_acc = acc
                if imputation not in results["data"][str(th_p)]:
                    results["data"][str(th_p)][imputation] =[acc]
                else:
                    results["data"][str(th_p)][imputation].append(acc)
                json.dump(results, open("imputation_results.json","w"))

        
