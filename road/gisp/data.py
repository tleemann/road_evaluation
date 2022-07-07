from cgi import test
import pdb
import os
import random
import imghdr
from PIL import Image
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm

DENSE_DATASETS = ['hapt', 'statlog', 'diabetes', 'heart', 'arthritis', 
        'synthesized4', 'synthesized8', 'synthesized1', 'ptbdb', 'mitbih', 'olivetti', 
        'credit', 'higgs', 'higgs_small', 'isolet', 'mnistdense', 'tox21/NR.AhR_norm',
        'nhanes/cholesterol_norm','nhanes/hypertension_norm']
DATASET_CLASSES = {'cifar10':10, 'mnist':10, 'hapt':5, 'statlog':6, 
        'diabetes':3, 'heart':2, 'arthritis':2, 'isolet':26,
        'synthesized4':2, 'synthesized8':2, 
        'ptbdb':2, 'mitbih':5, 'olivetti':40, 'tox21/NR.AhR_norm':2,
        'credit':2, 'higgs':2, 'higgs_small':2, 'mnistdense':10, 
        'nhanes/cholesterol_norm':2, 'nhanes/hypertension_norm':2}

class MissingProcess(object):
    def __init__(self, args):
        self.args = args
        
    def __call__(self, x):
        # select the generator
        if self.args.missing_type == 'natural':
            return self.natural(x)
        elif self.args.missing_type == 'mcar_uniform':
            # required args: missing_rate
            return self.mcar_uniform(x)
        elif self.args.missing_type == 'mcar_rect':
            # required attributes: beta1, beta2
            z = np.array([-4.76059049e+08,  1.41767280e+09, -1.84289709e+09,  1.37308572e+09,
                       -6.47409661e+08,  2.01326235e+08, -4.17706507e+07,  5.72632037e+06,
                              -5.03558055e+05,  2.70205976e+04, -8.32056957e+02,  1.47098602e+01])
            f = np.poly1d(z)
            if self.args.missing_rate <= 0.18:
                self.beta1 = 1.0
                self.beta2 = f(self.args.missing_rate)
            else:
                self.beta1 = f(self.args.missing_rate)
                self.beta2 = 1.0
            return self.mcar_rect(x)
        elif self.args.missing_type == 'mcar_rectinv':
            # required attributes: beta1, beta2
            z = np.array([-4.76059049e+08,  1.41767280e+09, -1.84289709e+09,  1.37308572e+09,
                       -6.47409661e+08,  2.01326235e+08, -4.17706507e+07,  5.72632037e+06,
                              -5.03558055e+05,  2.70205976e+04, -8.32056957e+02,  1.47098602e+01])
            f = np.poly1d(z)
            if (1.0-self.args.missing_rate) <= 0.18:
                self.beta1 = 1.0
                self.beta2 = f(1.0 - self.args.missing_rate)
            else:
                self.beta1 = f(1.0 - self.args.missing_rate)
                self.beta2 = 1.0
            return self.mcar_rectinv(x)
        elif self.args.missing_type == 'mnar_foreground':
            # required args: missing_rate
            return self.mnar_foreground(x)
        elif self.args.missing_type == 'mnar_background':
            # required args: missing_rate
            return self.mnar_background(x)
        else:
            raise NotImplementedError

    def natural(self, x_m):
        # fingerprint the input and use it as seed
        device = x_m.device
        mask = 1.0 - torch.isnan(x_m)
        x_i = torch.where(mask.bool(), x_m, torch.tensor((0.0), device=device))
        return x_i, x_m

    def mcar_uniform(self, x):
        # fingerprint the input and use it as seed
        seed = hash(x.numpy().tostring())
        seed = seed % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = x.device
        missing_rate_use = torch.rand(1)*self.args.missing_rate
        mask = (1 - torch.bernoulli(torch.ones(*x.shape[1:]) * missing_rate_use)).unsqueeze(0).expand(*x.shape)
        x_masked = torch.where(mask.bool(), x, torch.tensor(float('nan'), device=device))
        return x, x_masked

    def mcar_rect(self, x):
        # fingerprint the input and use it as seed
        seed = hash(x.numpy().tostring())
        seed = seed % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = x.device
        n_y = x.shape[2]
        n_x = x.shape[1]

        w = int(np.random.beta(self.beta1, self.beta2) * n_x)
        h = int(np.random.beta(self.beta1, self.beta2) * n_y)
        px = int(np.random.uniform(0.0,1.0) * n_x)
        py = int(np.random.uniform(0.0,1.0) * n_y)

        p1x = np.clip(px - w//2, 0, n_x)
        p1y = np.clip(py - h//2, 0, n_y)
        p2x = np.clip(px + w//2, 0, n_x)
        p2y = np.clip(py + h//2, 0, n_y)
        
        mask = torch.ones_like(x)
        mask[:, p1x:p2x, p1y:p2y] = 0.0
        x_masked = torch.where(mask.bool(), x, torch.tensor(float('nan'), device=device))
        return x, x_masked
    
    def mcar_rectinv(self, x):
        # fingerprint the input and use it as seed
        seed = hash(x.numpy().tostring())
        seed = seed % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = x.device
        n_y = x.shape[2]
        n_x = x.shape[1]

        w = int(np.random.beta(self.beta1, self.beta2) * n_x)
        h = int(np.random.beta(self.beta1, self.beta2) * n_y)
        px = int(np.random.uniform(0.0,1.0) * n_x)
        py = int(np.random.uniform(0.0,1.0) * n_y)

        p1x = np.clip(px - w//2, 0, n_x)
        p1y = np.clip(py - h//2, 0, n_y)
        p2x = np.clip(px + w//2, 0, n_x)
        p2y = np.clip(py + h//2, 0, n_y)
        
        mask = torch.ones_like(x)
        mask[:, p1x:p2x, p1y:p2y] = 0.0
        x_masked = torch.where((1.0-mask).bool(), x, torch.tensor(float('nan'), device=device))
        return x, x_masked
    
    def mnar_foreground(self, x):
        # fingerprint the input and use it as seed
        seed = hash(x.numpy().tostring())
        seed = seed % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = x.device
        mask_sel = (x>0.5).float()
        #mask = 1 - torch.bernoulli(mask_sel * self.args.missing_rate + \
        #        (1 - mask_sel) * (1-self.args.missing_rate))
        mask = 1 - torch.bernoulli(mask_sel * self.args.missing_rate + \
                (1 - mask_sel) * 0.1)
        x_masked = torch.where(mask.bool(), x, torch.tensor(float('nan'), device=device))
        return x, x_masked
    
    def mnar_background(self, x):
        # fingerprint the input and use it as seed
        seed = hash(x.numpy().tostring())
        seed = seed % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = x.device
        mask_sel = (x<0.5).float()
        #mask = 1 - torch.bernoulli(mask_sel * self.args.missing_rate)
        mask = 1 - torch.bernoulli(mask_sel * self.args.missing_rate + \
                (1 - mask_sel) * 0.1)
        x_masked = torch.where(mask.bool(), x, torch.tensor(float('nan'), device=device))
        return x, x_masked
    

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index, color_format='RGB'):
        img = Image.open(self.imgpaths[index])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        else:
            return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False, allowed_formats=None):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, dirs, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if self.__is_imgfile(path) == False:
                    continue
                imgpaths.append(path)
        imgpaths.sort()
        return imgpaths


class DenseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train, transform=None):
        super(DenseDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        # load the data file
        if train:
            path = self.data_dir + 'train.pkl'
        else:
            path = self.data_dir + 'test.pkl'
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return self.dataset['features'].shape[0]

    def __getitem__(self, index):
        x = self.dataset['features'][index,:]
        y = self.dataset['targets'][index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

### add dataset for food101
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        super(FoodDataset, self).__init__()
        # self.data_dir = os.path.expanduser(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.imgpaths = self.__get_img_path(self.data_dir, train)
        self.class_dict = self.__label2idx(self.data_dir)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index):
        label, img_path = self.imgpaths[index]
        target = self.class_dict[label]

        img_path = os.path.join(self.data_dir, 'images', '%s.jpg' % (img_path,))
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        return img_tensor, target

    def __label2idx(self, root):
        classes = {}
        path = os.path.join(root, 'meta', 'classes.txt')
        file = open(path, 'r')
        for i, line in enumerate(file.readlines()):
            classes[line.strip('\n')] = i
        return classes

    def __get_img_path(self, root, train):
        img_path_list = []
        if train:
            f = open(os.path.join(root, 'meta', 'train.txt'), 'r').readlines()
        else:
            f = open(os.path.join(root, 'meta', 'test.txt'), 'r').readlines()
        for line in f:
            label = line.split('/')[0]
            path = line.strip('\n')
            img_path_list.append((label, path))
        return img_path_list

class transform_randflip:
    def __init__(self):
        pass
    def __call__(self, x_x_m):
        np.random.seed(None)
        if np.random.rand() > 0.5:
            return (torch.flip(x_x_m[0], [2]), torch.flip(x_x_m[1], [2]))
        else:
            return x_x_m


class transform_np2tensor:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float)


class transform_addnoise:
    def __init__(self, std):
        self.std = std

    def __call__(self, x_x_m):
        if self.std != 0:
            x_x_m[1].add_(self.std * torch.randn_like(x_x_m[1]))
        return x_x_m

def load_dataset(args, post_transform):
    if args.dataset == 'mnist':
        trnsfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            post_transform,
            transform_addnoise(args.noise_std),
            ])  
        trnsfm_tst = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            post_transform,
            ])  
        train_dset = torchvision.datasets.MNIST(root=args.data_dir, train=True,
                                                        download=True, transform=trnsfm)
        test_dset = torchvision.datasets.MNIST(root=args.data_dir, train=False,
                                                       download=True, transform=trnsfm_tst)
    elif args.dataset == 'cifar10':
        trnsfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            post_transform,
            transform_addnoise(args.noise_std),
            transform_randflip()
            ])  
        trnsfm_tst = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            post_transform,
            ])  
        train_dset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                                        download=True, transform=trnsfm)
        test_dset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                                       download=True, transform=trnsfm_tst)
    elif args.dataset == 'celeba':
        trnsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            post_transform,
            transform_addnoise(args.noise_std),
            transform_randflip()
            ])  
        trnsfm_tst = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            post_transform,
            ])  
        train_dset = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm)
        test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm_tst)
    
    elif args.dataset in DENSE_DATASETS:
        trnsfm = torchvision.transforms.Compose([
            transform_np2tensor(),
            post_transform,
            transform_addnoise(args.noise_std),
            ])  
        trnsfm_tst = torchvision.transforms.Compose([
            transform_np2tensor(),
            post_transform,
            ])  
        train_dset = DenseDataset(args.data_dir, train=True, transform=trnsfm)
        test_dset = DenseDataset(args.data_dir, train=False, transform=trnsfm_tst)

    elif args.dataset == 'food-101':
        trnsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            post_transform,
            transform_addnoise(args.noise_std),
            transform_randflip()
            ])  
        trnsfm_tst = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            post_transform,
            ])
        train_dset = FoodDataset(args.data_dir, train=True, transform=trnsfm)
        test_dset = FoodDataset(args.data_dir, train=False, transform=trnsfm_tst)   
    
    return (train_dset, test_dset)
