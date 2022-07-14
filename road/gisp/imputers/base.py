import pdb
import os
import argparse
import time
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision

import utils
import models.gsmv
import models.resnet
import data


torch.set_num_threads(4)

class Imputer:
    def __init__(self):
        self.model = None
    
    def __call__(self, x):
        raise NotImplementedError

    def to(self, x):
        raise NotImplementedError

    def save(path):
        torch.save(self, path)
    
    def load(path):
        obj = torch.load(path, map_location='cpu')
        self.__dict__.update(obj.__dict__)
        return self
    
    def train(self, train_loader, test_loader, mfv, args):
        raise NotImplementedError

