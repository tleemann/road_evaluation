import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from resnet import resnet18
import matplotlib

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
from utils import *
import configs

def main():

	## read configs
	args = configs.arg_parse()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	input_path = args.input_path
	save_path = args.save_path
	device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
	batch_size = args.batch_size
	model_path = args.model_path

	# Directory to save the explanations
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	expl_str = args.expl_method
	save_train_path = os.path.join(save_path, expl_str.split('_')[0])
	save_test_path = os.path.join(save_path, expl_str.split('_')[0])
	if not os.path.isdir(save_train_path):
		os.makedirs(save_train_path)
	if not os.path.isdir(save_test_path):
		os.makedirs(save_test_path)
	save_train_file = '%s_train.pkl'%(expl_str.split('_')[1] if '_' in expl_str else 'base')
	save_test_file = '%s_test.pkl'%(expl_str.split('_')[1] if '_' in expl_str else 'base')

	transform_train = transforms.Compose([# transforms.RandomHorizontalFlip(),
			 					   transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	transform_test = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	model = resnet18() ### set relu(inplace=False)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 10)
	model = model.to(device)
	model.load_state_dict(torch.load(model_path))
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	scheduler = MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

	trainset = torchvision.datasets.CIFAR10(root=input_path, train=True,
	download=True, transform=transform_train)

	testset = torchvision.datasets.CIFAR10(root=input_path, train=False,
	download=True, transform=transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	  shuffle=True, num_workers=2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	 shuffle=False, num_workers=2)

	## get the acc of this model
	model.eval()
	correct = 0
	with torch.no_grad():
		for data in tqdm(testloader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == labels).sum().item()

		print('Accuracy of the network on test images: %.4f %%' % (
		 			100 * correct / len(testloader.dataset)))

	cifar_train_expl = {}

	## get explanation function
	get_expl = explanation_method(expl_str)
	for i_num in tqdm(range(len(trainset)), position=0, leave=True):
		expl_dict = {}
		sample, clss = trainset[i_num]
		outputs = model(sample.unsqueeze(0).to(device))
		_, predicted = torch.max(outputs.data, 1)
		expl = get_expl(model, sample.unsqueeze(0).to(device), clss)
		# print(predicted.data[0].cpu().numpy(), outputs.data[0].cpu().numpy(), clss)
		expl_dict['expl'] = expl
		expl_dict['prediction'] = predicted.data[0].cpu().numpy()
		expl_dict['label'] = clss
		expl_dict['predict_p'] = outputs.data[0].cpu().numpy()
		cifar_train_expl[i_num] = expl_dict
		# imshow_expl(expl)
		# _ = viz.visualize_image_attr(expl, sample, method="blended_heat_map",sign="all",
		# 								show_colorbar=True, title="Overlayed Integrated Gradients")

		# imshow(sample)
		# exit()
	save_dict(cifar_train_expl, save_train_path, save_train_file)
	# # show images
	# images = cifar_train_expl[0]
	# imshow(torch.from_numpy(images))

	cifar_test_expl = {}
	for i_num in tqdm(range(len(testset)), position=0, leave=True):
		expl_dict = {}
		sample, clss = testset[i_num]
		outputs = model(sample.unsqueeze(0).to(device))
		_, predicted = torch.max(outputs.data, 1)
		expl = get_expl(model, sample.unsqueeze(0).to(device), clss)
		# print(predicted.data[0].cpu().numpy(), outputs.data[0].cpu().numpy(), clss)
		expl_dict['expl'] = expl
		expl_dict['prediction'] = predicted.data[0].cpu().numpy()
		expl_dict['label'] = clss
		expl_dict['predict_p'] = outputs.data[0].cpu().numpy()
		cifar_test_expl[i_num] = expl_dict
	save_dict(cifar_test_expl, save_test_path, save_test_file)

if __name__ == '__main__':
	main()

