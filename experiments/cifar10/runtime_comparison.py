## Compare the runtime of the different approaches.
## pass two arguments python3 runtime_comparision.py {fixed, linear, gan} {retrain, noretrain} 
import torch
import torchvision
from torchvision import models as models
import torchvision.transforms as transforms
from torch import nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from road import run_road, use_device
from road.retraining import retraining
from road.imputations import NoisyLinearImputer, ChannelMeanImputer, ZeroImputer, GAINImputer
from road.utils import load_expl

# Config
use_device_gain = "cuda:1"

expl_path_test = "./data/ig/base_test.pkl"
expl_path_train = "./data/ig/base_train.pkl"

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


def get_imputer(imputation, retrain):
    """ Return the imputer object. """
    if imputation == "linear":
        imputer = NoisyLinearImputer(noise=0.01)
    elif imputation == "fixed":
        imputer = ChannelMeanImputer()
    elif imputation == "gan":
        imputer = GAINImputer("../../road/gisp/models/cifar_10_best.pt", use_device=(use_device_gain))
    else:
        raise ValueError(f"Invalid imputation {imputation}")
    return imputer

def run_no_retrain(imputation):
    _, explanation_test, _, prediction_test = load_expl(None, expl_path_test)
    # Load the model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(use_device)
    # load trained classifier
    model.load_state_dict(torch.load('../../data/cifar_8014.pth', map_location=use_device))

    # This transform has to be performed to run this model.
    transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_tensor = torchvision.transforms.Compose([transforms.ToTensor()])
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_tensor)
    cifar_test= torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_tensor)
    imputation_obj = get_imputer(imputation, False)
    accuracies, probs = run_road(model, cifar_test, explanation_test, transform_test, percentages, morf=True, batch_size = 32, imputation=imputation_obj)

def run_retrain(imputation):
    model = models.resnet18
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_tensor)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_tensor)
    ## set transforms
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    explanation_train, explanation_test, prediction_train, prediction_test = load_expl(expl_path_train, expl_path_test)
    imputation_obj = get_imputer(imputation, False)

    ## start retraining
    for percentage in percentages:
        res_acc, prob_acc = retraining(cifar_train, cifar_test, explanation_train, explanation_test, prediction_train, prediction_test, \
            [percentage], 10, model, transform_train=transform_train, transform_test=transform_test, epoch=40, morf=True, batch_size=64, imputation=imputation_obj)
    print("Result:", res_acc)

print(sys.argv)
if len(sys.argv) < 3:
    print("Please pass two arguments: python runtime_comparision.py <imputation> <retraining>")

imputation = sys.argv[1]
retrain = (sys.argv[2] == "retrain")
print(f"Running retrain={retrain} with imputation={imputation}")


if retrain:
    run_retrain(imputation)
else:
    run_no_retrain(imputation)
