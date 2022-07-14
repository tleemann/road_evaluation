import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from resnet import resnet50
import matplotlib
from torch.utils.data import DataLoader
import time

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
    save_expl_path = os.path.join(save_path, expl_str) if '_' in expl_str else os.path.join(save_path, '%s_base'%expl_str)

    if not os.path.isdir(os.path.join(save_expl_path, 'explanation', 'train')):
        os.makedirs(os.path.join(save_expl_path, 'explanation', 'train'))
    if not os.path.isdir(os.path.join(save_expl_path, 'prediction', 'train')):
        os.makedirs(os.path.join(save_expl_path, 'prediction', 'train'))
    if not os.path.isdir(os.path.join(save_expl_path, 'explanation', 'test')):
        os.makedirs(os.path.join(save_expl_path, 'explanation', 'test'))
    if not os.path.isdir(os.path.join(save_expl_path, 'prediction', 'test')):
        os.makedirs(os.path.join(save_expl_path, 'prediction', 'test'))

    # Food101
    model = resnet50()
    num_of_classes = 101

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_of_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    transform_train = transforms.Compose([transforms.Resize((224, 224)), # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    trainset = Data_Loader(root=input_path, train=True, dataset='Food-101', transform=transform_train)
    testset = Data_Loader(root=input_path, train=False, dataset='Food-101', transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print('Trainset: {}'.format(len(trainloader.dataset)))
    print('Testset: {}'.format(len(testloader.dataset)))

    ## get the acc of this model
    # model.half()
    model.eval()
    # correct = 0
    # with torch.no_grad():
    # 	for data in testloader:
    # 		inputs, labels, _ = data
    # 		inputs = inputs.to(device) #.to(device)
    # 		labels = labels.to(device) #.to(device)
    # 		outputs = model(inputs)
    # 		_, predicted = torch.max(outputs.data, 1)
    # 		correct += (predicted == labels).sum().item()
    #
    # 	print('Accuracy of the network on test images: %.4f %%' % (100 * correct / len(testloader.dataset)))



    ## get explanation function
    get_expl = explanation_method(expl_str)
    start = time.time()
    for i_num in tqdm(range(len(trainset))):
        sample, clss = trainset[i_num]
        sample = sample.unsqueeze(0).to(device) # .to(dtype=torch.half).to(device)
        outputs = model(sample)
        _, predicted = torch.max(outputs.data, 1)
        expl = get_expl(model, sample, clss)

        ### save expl and predictions
        np.save(os.path.join(save_expl_path, 'explanation', 'train', '%s.npy' % str(i_num)), expl)
        np.save(os.path.join(save_expl_path, 'prediction', 'train', '%s.npy' % str(i_num)), predicted.data[0].cpu().numpy())
    end = time.time() - start
    print('Explanation for Trainset complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))

    start = time.time()
    for i_num in tqdm(range(len(testset))):
        sample, clss = testset[i_num]
        sample = sample.unsqueeze(0).to(device) # .to(dtype=torch.half).to(device)
        outputs = model(sample)
        _, predicted = torch.max(outputs.data, 1)
        expl = get_expl(model, sample, clss)  # half precision torch.convert(dtype=float16)

        ### save expl and predictions
        np.save(os.path.join(save_expl_path, 'explanation', 'test', '%s.npy' % str(i_num)), expl)
        np.save(os.path.join(save_expl_path, 'prediction', 'test', '%s.npy' % str(i_num)), predicted.data[0].cpu().numpy())

    end = time.time() - start
    print('Explanation for Testset complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))

if __name__ == '__main__':

    main()

