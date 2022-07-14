import pickle 
import numpy as np
import torch

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


def load_dict(filename):
	with open(filename, 'rb') as f:
		di = pickle.load(f)
	return di

def load_expl(train_file, test_file):
	expl_train = []
	expl_test = []
	prediction_train = []
	prediction_test = []
	if train_file is not None:
		train_dict = load_dict(train_file)
		for i in range(len(train_dict)):
			# imshow_expl(train_dict[i]['expl'])
			expl_train.append(train_dict[i]['expl'])
			prediction_train.append(train_dict[i]['prediction'])
	if test_file is not None:
		test_dict = load_dict(test_file)
		for i in range(len(test_dict)):
			expl_test.append(test_dict[i]['expl'])
			prediction_test.append(test_dict[i]['prediction'])
	return expl_train, expl_test, prediction_train, prediction_test


def normalize_map(s):
    """ Min max normalization """
    epsilon = 1e-5
    norm_s = (s -np.min(s))/(np.max(s)-np.min(s) + epsilon)
    return norm_s

def rescale_channel(exp):
    exp = np.sum(exp, axis=-1)
    exp = normalize_map(exp)
    return exp