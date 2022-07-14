### Test how much class information is leaked by the masks.
from torch.utils.data.dataset import TensorDataset

from tqdm import tqdm
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import road
from road.utils import rescale_channel, load_expl

class Modified_Dataset(torchvision.datasets.CIFAR10):
	""" A modified CIFAR-10 dataset that also returns the corresponding explanantion masks. """
	def __init__(
	 		self,
	 		root,
			train=True,
			mask=None,
			th_p=1.0,
			remove=True,
			transform = None,
			target_transform = None,
			download = False,
			prediction = [],
			mask_baseline = None,
	) -> None:
		super().__init__(root, train, transform=transform, target_transform=target_transform, download=download)
		self.img_mask = mask
		self.th_p = th_p
		self.remove = remove
		self.use_baseline = mask_baseline
		self.prediction = prediction
		self.random_v = 1e-4*(np.random.randn(32,32))

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		_, target = datasets.CIFAR10.__getitem__(self, index)
		explanation = self.img_mask[index]
		mask_copy = rescale_channel(explanation).flatten()
		#print(mask_copy.shape, mask_copy[:5,:5])
		mask_copy = torch.tensor(mask_copy)
		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		# img = Image.fromarray(img)
		width, height = 32, 32
		salient_order = torch.argsort(mask_copy, axis=0, descending=True) # highest values first.
		#print(salient_order[:20])
		bitmask = torch.ones(width*height, dtype=torch.uint8) # Set to zero if pixel is removed.

		if self.remove:
			coords = salient_order[:int(width*height*self.th_p)]
			mean = 1-self.th_p
		else:
			coords = salient_order[int(width*height*(self.th_p)):]
			mean = self.th_p
			#print(len(coords))
		bitmask[coords] = 0
		bitmask = bitmask.reshape(1, width, height).repeat(3, 1, 1)
		#print(bitmask.shape)
		
		return bitmask-mean, target, 0

def train_val_net(model, trainloader, testloader, criterion, optimizer, scheduler, epoch, SAVE=False):
	""""
	Train and test function.
	Parameter:
		model: model to train, e.g. ResNet18
		trainloader, testloader: data loaders 
		criterion: loss function
		optimizer: optimizer
		scheduler: learning rate decay schedule
		epoch: training epoch number
		SAVE: set True to save the model
	"""
	model= model.to(road.use_device)
	best_acc = 0.0
	best_prob = 0.0
	for epoch in range(epoch):  # loop over the dataset multiple times
		running_loss = 0.0
		correct = 0
		model.train()
		for i, data in enumerate(tqdm(trainloader)):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels, predictions = data
			inputs = inputs.to(road.use_device)
			labels = labels.to(road.use_device)
			predictions = predictions.to(road.use_device)

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
		print('Training [%d] loss: %.5f; acc: %.5f' %
		  (epoch + 1, running_loss / len(trainloader.dataset), correct / len(trainloader.dataset)))
		scheduler.step()
		## eval the model
		correct = 0
		prob = 0.0
		model.eval()
		with torch.no_grad():
			for data in tqdm(testloader):
				inputs, labels, predictions = data
				inputs = inputs.to(device=road.use_device)
				labels = labels.to(device=road.use_device)
				predictions = predictions.to(device=road.use_device)
				outputs = model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				correct += (predicted == labels).sum().item()
			##### calculate the average probability
				probs = torch.nn.functional.softmax(outputs, dim=1)
				for k,p in enumerate(predictions):
					prob += probs[k,p].cpu().numpy()

			print('Accuracy of the network on test images: %.4f %%, average probability:  %.4f' % (
			 			100 * correct / len(testloader.dataset), prob / len(testloader.dataset)))
		if best_acc < (correct/len(testloader.dataset)):
			best_acc = correct / len(testloader.dataset)
			if SAVE:
				PATH = './cifar_net.pth'
				torch.save(model.state_dict(), PATH)
		if best_prob < (prob/len(testloader.dataset)):
			best_prob = prob / len(testloader.dataset)
			if SAVE:
				PATH = './cifar_net.pth'
				torch.save(model.state_dict(), PATH)
	return best_acc, best_prob

def run_benchmark(maintype="ig", dtype="sq"):
	""" Run the leakage test. """
	explanation_train, explanation_test, prediction_train, prediction_test \
		= load_expl('./data/%s/%s_train.pkl'%(maintype, dtype), './data/%s/%s_test.pkl'%(maintype, dtype))

	#explanations_tensor = torch.stack([torch.Tensor(rescale_channel(expl)) for expl in explanation_train])
	#print(explanations_tensor.shape)

	for roar_flag in [True, False]:
		batch_size = 32
		result_l = []
		for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
			trainset = Modified_Dataset(root='./data', train=True, mask=explanation_train, th_p=p, remove=roar_flag,
			download=True, prediction=prediction_train)


			testset = Modified_Dataset(root='./data', train=False, mask=explanation_test, th_p=p, remove=roar_flag, 
			download=True, prediction=prediction_test)

			
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
			shuffle=True, num_workers=1)

			testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
			shuffle=False, num_workers=1)
			
			# plt.ioff()
			#print(testset[0])
			#imshow(testset[0][0])
			#imshow(testset[2336][0], "test", 2336)

			#imshow(trainset[0][0], "train", 0)
			#imshow(trainset[2336][0], "train", 2336)
			# classes = ('plane', 'car', 'bird', 'cat',
			#    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

			# # get some random training images
			# dataiter = iter(testloader)
			# images, labels = dataiter.next()

			# # show images
			# image_list.append(images[2])
			# # print labels
			# #print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
			model = models.resnet18(pretrained=False)
			num_ftrs = model.fc.in_features
			# Here the size of each output sample is set to 2.
			# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
			model.fc = nn.Linear(num_ftrs, 10)
			model = model.cuda()
			criterion = nn.CrossEntropyLoss().cuda()
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			scheduler = MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

			res, probs = train_val_net(model, trainloader, testloader, criterion, optimizer, scheduler, 40)
			result_l.append(res)
			roar_text = "roar" if roar_flag else "kar"	
		with open('./data/mask_only/%s_%s_%s.txt'%(maintype, roar_text, dtype),'a') as f:
			for item in result_l:
				f.write(str(item) + ' ')
			f.write("\n")


if __name__ == '__main__':
	for tpye in ['base', 'sg', 'sq', 'var']:
		run_benchmark(maintype="ig", dtype=tpye)

	for tpye in ['base', 'sg', 'sq', 'var']:
		run_benchmark(maintype="gb", dtype=tpye)
