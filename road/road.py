import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import typing as tp

from .imputations import BaseImputer, ChannelMeanImputer, NoisyLinearImputer, GAINImputer, _from_str
from .imputed_dataset import ImputedDataset, ImputedDatasetMasksOnly
from .gpu_dataloader import ImputingDataLoaderWrapper
from .retraining import road_eval

def run_road(model, dataset_test, explanations_test, transform_test, percentages, morf=True, batch_size=64, imputation = NoisyLinearImputer(noise=0.01)):
    """ Run the ROAD benchmark. 
        model: Pretrained model on data set.
        dataset_test: The test set to run the benchmark on. Should deterministically return a (tensor, tensor)-tuple for each index.
        explanations_test: Attributions for each data point. List or array (or any SupportsIndex) with same len as dataset_test.
        transform_test: Transforms to be applied on the Modified data set after the imputation, e.g. normalization. Transformations applied before imputation should
            already be incorporated into dataset_test.
        percentages: List of percentage values that will be tested.
        morf: True, if morf oder should be applied, else false.
        batch_size: Batch size to use for the benchmark. Can be larger as it does inference only.
        imputation: Either an Imputer object (Subclass of imputations.BaseImputer) or a string in {linear, fixed, zero}.
    """
    # Construct Imputer from string.
    if type(imputation) == str:
        imputation == _from_str(imputation)

    res_acc = torch.zeros(len(percentages))
    prob_acc = torch.zeros(len(percentages))
    for i, p in enumerate(percentages):
        print("Running evaluation for p=", p)
        if type(imputation) == GAINImputer and imputation.run_on_device != "cpu":
            print("Using GPU-Accelerated DataLoader for GAIN.")
            ds_test_imputed_lin = ImputedDatasetMasksOnly(dataset_test, mask=explanations_test, th_p=p, remove=morf,
                                                prediction=None, use_cache=False)
            base_testloader = torch.utils.data.DataLoader(ds_test_imputed_lin, batch_size=batch_size, shuffle=False,
                                                    num_workers=4)
            testloader =  ImputingDataLoaderWrapper(base_testloader, imputation, image_transform=transform_test)          
        else:
            ds_test_imputed_lin = ImputedDataset(dataset_test, mask=explanations_test, th_p=p, remove=morf, imputation = imputation, 
                    transform = transform_test, target_transform = None, prediction = None, use_cache=False)
            testloader = torch.utils.data.DataLoader(ds_test_imputed_lin, batch_size=batch_size, shuffle=False, num_workers=8)
            
        print(len(ds_test_imputed_lin), len(testloader))
        acc_avg, prob_avg = road_eval(model, testloader)
        res_acc[i] = acc_avg
        prob_acc[i] = prob_avg
    return res_acc, prob_acc