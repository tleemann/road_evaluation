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
from .imputations import BaseImputer, ChannelMeanImputer, NoisyLinearImputer
from .road import normalize_map, rescale_channel
class ImputedDatasetMasksOnly(torch.utils.data.Dataset):
	"""
	Base class for an imputed dataset according to the ROAD benchmark.
	Unlike the dataset in the original method, the imputation is left to the 
	data_loader.
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
		transforms: transform functions for image
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
		self.use_cache = use_cache
		self.cached_img = {}
		self.cached_target = {}
		self.cached_pred = {}
		self.cached_mask = {}

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

			if self.use_cache: # Add to cache.
				self.cached_img[index] = img
				self.cached_target[index] = target
				self.cached_pred[index] = pred
				self.cached_mask[index] = bitmask
		else:	
			img = self.cached_img[index]
			target = self.cached_target[index]
			pred = self.cached_pred[index]

		return img, target, pred, bitmask

	def __len__(self):
		return len(self.base_dataset)


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