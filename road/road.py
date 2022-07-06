import torch
from torch._C import device
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import typing as tp


from .imputations import BaseImputer, ChannelMeanImputer, NoisyLinearImputer, GAINImputer, _from_str
from .gpu_dataloader import ImputedDatasetMasksOnly, ImputingDataLoaderWrapper

## Define default device for ROAD
use_device="cpu"
if torch.cuda.is_available():
    use_device="cuda:0"

torch.manual_seed(2)
np.random.seed(2)

## Some utility functions.
### set device, use cuda if available

def set_device(device_str):
    """ Change device. """
    import road # Oh Python...
    road.use_device = device_str

def normalize_map(s):
    """ Min max normalization """
    epsilon = 1e-5
    norm_s = (s -np.min(s))/(np.max(s)-np.min(s) + epsilon)
    return norm_s

def rescale_channel(exp):
    exp = np.sum(exp, axis=-1)
    exp = normalize_map(exp)
    return exp


class ImputedDataset(torch.utils.data.Dataset):
    """
    Base class for an imputed dataset according to the ROAD benchmark.
    Parameters:
        base_dataset: The original dataset. Must return an (image, label) tuple 
            if base_dataset[index] is called, where both are tensors. 
            Please make sure the return value is deterministic.
        mask: explanation maps. The explanation maps to used. Must return a torch.tensor
            of the same size as image in base_dataset when mask[index] is called.
            Can be a list, torch.utils.data.Dataset, as long as the index function is defined.
            Must match length of base_dataset.
        th_p: percentage of pixels to be pertubed (0.0-1.0)
        remove: if True, MoRF oder is applied, else LeRF
        imputation: An imputation module. See class BaseImputer for documentation
            of the interface
        transforms: transform functions for image, that are applied after imputation.
        target_transform: transform function for labels
        prediction: predictions made when computing the explanation maps
        use_cache: whether to cache the imputated data set (may be useful
            if the imputed dataset is used for model retraining and the imputation 
            takes long to compute.)
    """
    def __init__(
             self,
            base_dataset, # : tp.Union[torch.utils.data.Dataset, tp.SupportsIndex],
            mask, #: tp.Union[torch.utils.data.Dataset, tp.SupportsIndex],
            th_p=1.0,
            remove=True,
            imputation: BaseImputer = ChannelMeanImputer(),
            transform = None,
            target_transform = None,
            prediction = [],
            use_cache=False
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.img_mask = mask
        self.th_p = th_p
        self.remove = remove
        self.prediction = prediction
        # a constant small perturbation for attribution map with many equal values.
        self.random_v = 1e-4*(np.random.randn(*self.img_mask[0].shape[:2]))
        self.imputation = imputation # Either 'fixed' or 'linear'
        self.use_cache = use_cache
        self.cached_img = {}
        self.cached_target = {}
        self.cached_pred = {}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not self.use_cache or index not in self.cached_img:
            img, target = self.base_dataset[index]
            pred = self.prediction[index] if self.prediction else 0
            explanation = self.img_mask[index]
            mask_copy = rescale_channel(explanation)
            mask_copy += self.random_v
            mask_copy = mask_copy.reshape(-1,1)
            mask_copy = torch.tensor(mask_copy)
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # img = Image.fromarray(img)
            width, height = img.size(-2), img.size(-1)
            salient_order = torch.argsort(mask_copy, axis=0, descending=True) # highest values first.
            bitmask = torch.ones(width*height, dtype=torch.uint8) # Set to zero if pixel is removed.

            ## my modification
            if self.remove:
                coords = salient_order[:int(width*height*self.th_p)]
            else:
                coords = salient_order[int(width*height*(self.th_p)):]
                #print(len(coords))
            bitmask[coords] = 0
            bitmask = bitmask.reshape(width, height)

            # Call the imputor.
            img = self.imputation(img, bitmask)
        

            if self.use_cache: # Add to cache.
                self.cached_img[index] = img
                self.cached_target[index] = target
                self.cached_pred[index] = pred
        else:
            img = self.cached_img[index]
            target = self.cached_target[index]
            pred = self.cached_pred[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, pred

    def __len__(self):
        return len(self.base_dataset)


# def run_road(model, dataset_test: tp.SupportsIndex, explanations_test: tp.SupportsIndex,
# 			 transform_test, percentages: tp.List[float], morf=True, batch_size = 64, imputation = NoisyLinearImputer(noise=0.01)):
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
    from .retraining import road_eval

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