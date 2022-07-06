## A quicker version of the Imputed Dataset, that allows to computute the imputations on the GPU (particularly important for GAIN)
import imp
import torch

import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
# from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
import typing as tp

from .utils import normalize_map, rescale_channel
from .imputations import BaseImputer, NoisyLinearImputer
from .imputed_dataset import ImputedDatasetMasksOnly

class ImputingDataLoaderWrapper(DataLoader):
	def __init__(self, org_data_loader: DataLoader, imputer: BaseImputer, image_transform=None, target_transform=None):
		""" Take a base data loader and do imputation on top.
			the image_transforms and target transforms are applied on top, so please make sure that they also work for batched
			images / labels.
		"""
		self.base_dl = org_data_loader
		self.imputer = imputer
		self.image_transform = image_transform
		self.target_transform = target_transform

	def __len__(self) -> int:
		return len(self.base_dl)

	def __iter__(self):
		class ImputingDLIter(tp.Iterator):
			def __init__(self, org_data_loader: DataLoader, imputer: BaseImputer, image_transform, target_transform):
				self.base_dl = org_data_loader
				self.imputer = imputer
				self.image_transform = image_transform
				self.target_transform = target_transform
				self.myiter = self.base_dl.__iter__()

			def __next__(self):
				""" Get an item from the base iterator """
				img, target, pred, bitmask = next(self.myiter) # Stop iteraton is passed on.
				img = self.imputer.batched_call(img, bitmask)
				#print(img.shape)
				if self.target_transform:
					target = self.target_transform(target)
				if self.image_transform:
					img = self.image_transform(img)
				return img, target, pred

		return ImputingDLIter(self.base_dl, self.imputer, self.image_transform, self.target_transform)
	
	@property
	def dataset(self):
		return self.base_dl.dataset
	
	@property
	def batch_size(self):
		return self.base_dl.batch_size

def run_road_batched(model, dataset_test, explanations_test, transform_test, percentages, morf=True, batch_size=64, imputation = NoisyLinearImputer(noise=0.01)):
	""" Run the ROAD benchmark. 
		model: Pretrained model on data set
		dataset_test: the test set to run the benchmark on. Should deterministically return a (tensor, tensor)-tuple.
		explanations_test: Attributions for each data point. List or array with same len as dataset_test.
		transform_test: Transforms to be applied on the Modified data set, e.g. normalization.
		percentages: List of percentage values that will be tested.
		morf: True, if morf oder should be applied, else false.
		batch_size: Batch size to use for the benchmark. Can be larger as it does inference only.
	"""
	from .retraining import road_eval
	res_acc = torch.zeros(len(percentages))
	prob_acc = torch.zeros(len(percentages))
	for i, p in enumerate(percentages):
		print("Running evaluation for p=", p)
		ds_test_imputed_lin = ImputedDatasetMasksOnly(dataset_test, mask=explanations_test, th_p=p, remove=morf, prediction = None, use_cache=False)
		testloader = torch.utils.data.DataLoader(ds_test_imputed_lin, batch_size=batch_size, shuffle=False, num_workers=8)
		gpu_loader = ImputingDataLoaderWrapper(testloader, imputer=imputation, image_transform=transform_test)
		print(len(ds_test_imputed_lin), len(gpu_loader))
		acc_avg, prob_avg = road_eval(model, gpu_loader)
		res_acc[i] = acc_avg
		prob_acc[i] = prob_avg
	return res_acc, prob_acc