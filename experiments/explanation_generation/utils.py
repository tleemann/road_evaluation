import torch
from torch._C import device
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
from captum.attr import GradientShap, DeepLift, IntegratedGradients, LayerGradCam, LayerAttribution, NoiseTunnel, GuidedBackprop


# ------------- Functions ----------------
def save_dict(dict, folder, filename):
	with open(os.path.join(folder, filename), 'wb') as f:
		pickle.dump(dict, f)

def load_dict(filename):
	with open(filename, 'rb') as f:
		di = pickle.load(f)
	return di

def load_expl(train_file, test_file):
	train_dict = load_dict(train_file)
	test_dict = load_dict(test_file)
	expl_train = []
	expl_test = []
	prediction_train = []
	prediction_test = []
	for i in range(50000):
		# imshow_expl(train_dict[i]['expl'])
		expl_train.append(train_dict[i]['expl'])
		prediction_train.append(train_dict[i]['prediction'])
	for i in range(10000):
		expl_test.append(test_dict[i]['expl'])
		prediction_test.append(test_dict[i]['prediction'])
	return expl_train, expl_test, prediction_train, prediction_test

def normalize_map(s):
	epsilon = 1e-5
	norm_s = (s -np.min(s))/(np.max(s)-np.min(s) + epsilon)
	# norm_s = (s -torch.min(s))/(torch.max(s)-torch.min(s) + epsilon)
	return norm_s

def imshow(img):
	# img = normalize_map(img)
	img = img / 2 + 0.5 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()
	

def imshow_expl(img):
	img = np.sum(img, axis=-1)
	img = normalize_map(img)
	# img = img / 2 + 0.5 # unnormalize
	# if len(img.shape)>2:
	# 	plt.imshow(np.transpose(img, (1, 2, 0)))
	# else:
	plt.imshow(img)
	plt.show()


def rescale_channel(exp):
	exp = np.sum(exp, axis=-1)
	exp = normalize_map(exp)
	# gray = 0.2989*exp[0,:,:] + 0.5870*exp[1,:,:] + 0.1140*exp[2,:,:]
	return exp


# ------------- Data Loader for CIFAR10-----------------
class Modified_Dataset(torchvision.datasets.CIFAR10):
	"""
	Dataset for CIFAR-10 with pertubations.
	Parameters:
		root: root path where the explanations stored
		train: set to True if is training set, otherwise, test set
		mask: explanation maps
		th_p: percentage of pixels pertubed
		imputation: linear or fixed value (mean values)
		transforms: transform functions
		target_transform: transform function for targets
		download: set True to downlaod CIFAR-10
		prediction: predictions made when computing the explanation maps
		mask_baseline: NOT used in this script, please set to None
		use_cache: to cache the imputated data set
	"""
	def __init__(
	 		self,
	 		root,
			train=True,
			mask=None,
			th_p=1.0,
			remove=True,
			imputation = "linear",
			transform = None,
			target_transform = None,
			download = False,
			prediction = [],
			mask_baseline = None,
			use_cache=False
	) -> None:

		super().__init__(root, train, transform=transform, target_transform=target_transform, download=download)

		self.img_mask = mask
		self.th_p = th_p
		self.remove = remove
		self.use_baseline = mask_baseline
		self.prediction = prediction
		self.random_v = 1e-4*(np.random.randn(32,32))
		self.imputation = imputation # Either 'fixed' or 'linear'
		self.use_cache = use_cache
		self.cached_img = {}
		self.cached_target = {}
		self.cached_pred = {}

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		if not self.use_cache:
			img, target = datasets.CIFAR10.__getitem__(self, index)
			pred = self.prediction[index]
			if self.use_baseline is None:
				explanation = self.img_mask[index]
				mask_copy = rescale_channel(explanation)
				mask_copy += self.random_v
				mask_copy = mask_copy.reshape(-1,1)
			else:
				mask_copy = self.use_baseline

			mask_copy = torch.tensor(mask_copy)
		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		# img = Image.fromarray(img)
			width, height = 32, 32
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
			if self.imputation == "fixed":
				for c in range(3):
					mean_c = img[c,:,:].mean()
					imgsubtensor = img[c,:,:]
					imgsubtensor[bitmask==0] = mean_c
			elif self.imputation == "linear":
				img = lin_infilling(img, bitmask, noise=0.0, sparse=True)
			else:
				raise ValueError(f"Unknown infilling strategy {self.imputation}.")

			if self.target_transform is not None:
				target = self.target_transform(target)
		else:
			img = self.cached_img[index]
			target = self.cached_target[index]
			pred = self.cached_pred[index]

		if self.use_cache:
			return img, target, pred
		else:
			return img, target, pred, index

	def set_use_cache(self, use_cache):
		if use_cache:
			# imgs = tuple(self.cached_data)
			# self.cached_img = torch.stack(self.cached_img)
			# self.cached_target = torch.Tensor(self.cached_target)
			# self.cached_pred = torch.Tensor(self.cached_pred)
			pass
		else:
			self.cached_img = []
			self.cached_target = []
			self.cached_pred = []
		self.use_cache = use_cache

# ------------- Functions in Data_Loader-----------------
def birdsnap_fail(root):
	f = open(os.path.join(root, 'download', 'fail2.txt'))
	fail_path_list = []
	for line in f.readlines():
		path = line.split()[1]
		fail_path_list.append(path)
	return fail_path_list

def get_img_path(root, dataset, train):
	img_path_list = []
	if dataset == 'Food-101':
		if train:
			f = open(os.path.join(root, 'meta', 'train.txt'), 'r').readlines()
		else:
			f = open(os.path.join(root, 'meta', 'test.txt'), 'r').readlines()
		for line in f:
			label = line.split('/')[0]
			path = line.strip('\n')
			img_path_list.append((label, path))
		return img_path_list

	elif dataset == 'Birdsnap':
		# fail_list = birdsnap_fail(root)
		f = open(os.path.join(root, 'images_low.txt'), 'r').readlines()
		img_test = []
		img_train = []
		test_file = open(os.path.join(root, 'test_images.txt'), 'r')
		img_test_path_list = [test_line.strip('\n').split('.')[0] for test_line in test_file.readlines()[1:]]
		for line in f:
			label = line.split('/')[0].lower()
			path = line.strip('\n')
			if path in img_test_path_list:
				img_test.append((label, path))
			else:
				img_train.append((label, path))
		if train:
			return img_train
		else:
			return img_test

def label2idx(root, dataset):
	classes = {}
	if dataset == 'Food-101':
		path = os.path.join(root, 'meta', 'classes.txt')
		file = open(path, 'r')
		for i, line in enumerate(file.readlines()):
			classes[line.strip('\n')] = i
	elif dataset == 'Birdsnap':
		path = os.path.join(root, 'species.txt')
		file = open(path, 'r')
		for line in file.readlines()[1:]:
			label = line.split()[-1]
			id = line.split()[0]
			classes[label] = int(id)
	return classes

def loader(path, transform):
	img = Image.open(path).convert("RGB")
	img_tensor = transform(img)
	return img_tensor

def load_npy(path):
	img = np.load(path)
	return img

# ------------- Data_Loader for Food-101 -----------------
class Data_Loader(Dataset):
	"""
	Dataset for reading Food-101 or other datasets.
	Parameters:
		root: root path where the explanations stored
		dataset: the name of dataset. Currently, we support Food-101 and Birdsap
		train: set to True if is training set, otherwise, test set
		transforms: transform functions
	"""
	def __init__(
			self,
			root,
			dataset,
			train=True,
			transform=None,
	) -> None:

		super().__init__()

		self.root = root # "../dataset/food-101/" or "../dataset/birdsnap/"
		self.train = train # train or test
		self.transform = transform
		self.dataset = dataset

		self.class_dict = label2idx(self.root, self.dataset)
		if self.dataset == 'Food-101':
			self.dataset_path = os.path.join(self.root, 'images')
			self.img_path_list = get_img_path(self.root, self.dataset, self.train)
		elif self.dataset == 'Birdsnap':
			self.dataset_path = os.path.join(self.root, 'download', 'images_low')
			self.img_path_list = get_img_path(self.root, self.dataset, self.train)
		else:
			raise ValueError("Unknown dataset.")

	def __len__(self):
		return len(self.img_path_list)

	def __getitem__(self, index):
		label, img_path = self.img_path_list[index]
		target = self.class_dict[label]
		img = loader(os.path.join(self.dataset_path, '%s.jpg' % (img_path,)), self.transform)

		target = target - 1 if self.dataset == 'Birdsnap' else target

		return img, target


# ------------- Functions for computating explanations -----------------
### In our paper, we used IG(), GB(), IG_SG(), GB_SG() for different variants by changing "nt_type": https://captum.ai/api/noise_tunnel.html

def attribute_image_features(model, algorithm, input, target, **kwargs):
	model.zero_grad()
	tensor_attributions = algorithm.attribute(input,
											  target=target,
											  **kwargs
											 )
	return tensor_attributions

def IG(model, sample, target):
	ig = IntegratedGradients(model)
	# baseline = torch.zeros(1,3,32,32).cuda()
	# attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
	attr_ig, delta = attribute_image_features(model, ig, sample, target, baselines=sample * 0, return_convergence_delta=True)
	attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
	return attribution

def IG_SG(model, sample, target, nt_type='smoothgrad'):
	ig = IntegratedGradients(model)
	nt = NoiseTunnel(ig)
	attr_ig_nt = attribute_image_features(model, nt, sample, target, baselines=sample * 0, nt_type=nt_type,
									  nt_samples=15, stdevs=0.2)
	attribution = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
	return attribution

def GB(model, sample, target):
	gb = GuidedBackprop(model)
	# baseline = torch.zeros(1,3,32,32).cuda()
	# attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
	attr_ig = attribute_image_features(model, gb, sample, target)
	attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
	return attribution

def GB_SG(model, sample, target, nt_type='smoothgrad'):
	gb = GuidedBackprop(model)
	nt = NoiseTunnel(gb)
	# baseline = torch.zeros(1,3,32,32).cuda()
	# attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
	attr_ig_nt = attribute_image_features(model, nt, sample, target, nt_samples=100, nt_type=nt_type)
	attribution = np.transpose(attr_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
	return attribution

def explanation_method(expl_str):
	""" Return a default explanation from a string. """
	def compute_explanation(model, sample, target):
		if expl_str == "ig":
			return IG(model, sample, target)
		elif expl_str == "gb":
			return GB(model, sample, target)
		elif expl_str == "ig_sg":
			return IG_SG(model, sample, target)
		elif expl_str == "gb_sg":
			return GB_SG(model, sample, target)
		elif expl_str == "ig_sq":
			return IG_SG(model, sample, target, nt_type='smoothgrad_sq')
		elif expl_str == "gb_sq":
			return GB_SG(model, sample, target, nt_type='smoothgrad_sq')
		elif expl_str == "ig_var":
			return IG_SG(model, sample, target, nt_type='vargrad')
		elif expl_str == "gb_var":
			return GB_SG(model, sample, target, nt_type='vargrad')
		else:
			raise ValueError("Unknown explanation string. Please use {ig, gb, ig_sg, gb_sg, ig_sq, gb_sq, ig_var, gb_var}.")
	return compute_explanation

### some other explanation functions
def GradShap_SG(model, sample, target, nt_type='vargrad'):
	gradient_shap = GradientShap(model)
	nt = NoiseTunnel(gradient_shap)
	attr_ig_nt = attribute_image_features(model, nt, sample, target, baselines=sample * 0,  nt_samples=100, nt_type=nt_type, stdevs=0.2)
	attribution = np.transpose(attr_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
	return attribution

def GradShap(model, sample, target):
	gradient_shap = GradientShap(model)
	attri_gs = attribute_image_features(model, gradient_shap, sample, target, baselines=sample*0, n_samples=1, stdevs=0.0)
	attribution = np.transpose(attri_gs.squeeze().cpu().detach().numpy(), (1, 2, 0))
	return attribution

def Deeplift(model, sample, target):
	deeplift = DeepLift(model)
	attribution = deeplift.attribute(sample.reshape(1,3,32,32).cuda(), target=target)
	return attribution.detach().squeeze().cpu().numpy()

def GradCAM(model, sample, target):
	sample = Variable(sample.unsqueeze(0)).cuda()
	layer_gc = LayerGradCam(model, model.layer2)
	attr= layer_gc.attribute(sample, target=target)
	upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))
	return upsampled_attr.detach().squeeze().cpu().numpy()