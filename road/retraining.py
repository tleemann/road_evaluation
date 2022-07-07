## Everything related to the retraining benchmark.
import torch
from torch._C import device
import torchvision.transforms as transforms
from torchvision import datasets, models
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
import typing as tp
import sys

from .imputations import BaseImputer, ChannelMeanImputer, GAINImputer, NoisyLinearImputer
from .utils import use_device
from .imputed_dataset import ImputedDataset, ImputedDatasetMasksOnly
from .gpu_dataloader import ImputingDataLoaderWrapper
# from tqdm import tqdm

def road_eval(model, testloader):
    # eval the model for a specific modified data set
    # Return accuracy and average true class probability.
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    with torch.no_grad():
        # for data in tqdm(testloader)
        for i, data in enumerate(testloader):
            inputs, labels, predictions = data
            inputs = inputs.to(use_device)
            labels = labels.to(use_device)
            predictions = predictions.to(use_device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            ##### calculate the average probability
            probs = torch.nn.functional.softmax(outputs, dim=1)
            for k,p in enumerate(predictions):
                prob += probs[k,p].cpu().numpy()
            #break
    print('Accuracy of the network on test images: %.4f %%, average probability:  %.4f' % (
                    100 * correct / len(testloader.dataset), prob / len(testloader.dataset)))
    acc_avg = correct / len(testloader.dataset)
    prob_avg = prob / len(testloader.dataset)
    return acc_avg, prob_avg

def road_train(model, trainloader, testloader, criterion, optimizer, epoch, scheduler=None):
    # eval the model for a specific modified data set
    # Return accuracy and average true class probability.
    model = model.to(use_device)
    best_acc = 0.0
    best_prob = 0.0
    for epoch in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        model.train()
        print('starting epoch %d'%(epoch+1), datetime.now())
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, predictions = data
            inputs = inputs.to(use_device)
            labels = labels.to(use_device)
            predictions = predictions.to(use_device)
            #print(inputs.shape, inputs.device, datetime.now())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            #break

        print('Training [%d] loss: %.5f; acc: %.5f' % (epoch + 1, running_loss / len(trainloader.dataset), correct / len(trainloader.dataset)))
        if scheduler is not None:
            scheduler.step()

        acc_avg, prob_avg = road_eval(model, testloader)
        if best_acc < acc_avg:
            best_acc = acc_avg
        if best_prob < prob_avg:
            best_prob = prob_avg
        print('--' * 20)

    return best_acc, best_prob

def retraining(dataset_train, dataset_test, explanations_train, explanations_test, predictions_train, predictions_test,
               percentages, num_of_classes, modelclass, transform_train=None, transform_test=None, epoch=40, morf=True, batch_size=64, 
               save_path=".", imputation=NoisyLinearImputer(noise=0.01)):
    """ Run the ROAR benchmark.
        modelclass: model class
        dataset_test: the test set to run the benchmark on. Should deterministically return a (tensor, tensor)-tuple.
        explanations_test: Attributions for each data point. List or array with same len as dataset_test.
        transform_test: Transforms to be applied on the Modified data set, e.g. normalization.
        percentages: List of percentage values that will be tested.
        morf: True, if morf oder should be applied, else false.
        batch_size: Batch size to use for the benchmark. Can be larger as it does inference only.
    """
    res_acc = torch.zeros(len(percentages))
    prob_acc = torch.zeros(len(percentages))
    for i, p in enumerate(percentages):
        model = modelclass()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        model = model.to(use_device)

        criterion = nn.CrossEntropyLoss().to(use_device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

        print("--" * 25)
        print("Running evaluation for p=", p, datetime.now())
        if type(imputation) == GAINImputer:
            print("Using GPU-Accelerated DataLoader for GAIN.")
            ds_train_imputed_lin = ImputedDatasetMasksOnly(dataset_train, mask=explanations_train, th_p=p, remove=morf,
                                                prediction=predictions_train, use_cache=False)
            ds_test_imputed_lin = ImputedDatasetMasksOnly(dataset_test, mask=explanations_test, th_p=p, remove=morf,
                                                prediction=predictions_test, use_cache=False)
            
            base_trainloader = torch.utils.data.DataLoader(ds_train_imputed_lin, batch_size=batch_size, shuffle=True,
                                                    num_workers=4)
            base_testloader = torch.utils.data.DataLoader(ds_test_imputed_lin, batch_size=batch_size, shuffle=False,
                                                    num_workers=4)

            trainloader =  ImputingDataLoaderWrapper(base_trainloader, imputation, image_transform=transform_train)  
            testloader =  ImputingDataLoaderWrapper(base_testloader, imputation, image_transform=transform_test)          
        else:
            ds_train_imputed_lin = ImputedDataset(dataset_train, mask=explanations_train, th_p=p, remove=morf,
                                                imputation=imputation, transform=transform_train,
                                                target_transform=None, prediction=predictions_train,
                                                use_cache=False)
            ds_test_imputed_lin = ImputedDataset(dataset_test, mask=explanations_test, th_p=p, remove=morf,
                                                imputation=imputation, transform=transform_test,
                                                target_transform=None, prediction=predictions_test,
                                                use_cache=False)
            trainloader = torch.utils.data.DataLoader(ds_train_imputed_lin, batch_size=batch_size, shuffle=True,
                                                    num_workers=16)
            testloader = torch.utils.data.DataLoader(ds_test_imputed_lin, batch_size=batch_size, shuffle=False,
                                                    num_workers=16)
        print("Trainloader:", len(dataset_train), len(ds_train_imputed_lin), len(trainloader.dataset))
        print("Testloader:", len(dataset_test), len(ds_test_imputed_lin), len(testloader.dataset))

        res, prob = road_train(model, trainloader, testloader, criterion, optimizer, epoch, scheduler)
        res_acc[i] = res
        prob_acc[i] = prob
        print('--' * 50)
        print('--' * 50)
        del model
        del optimizer
        del scheduler
        del criterion

    return res_acc, prob_acc
