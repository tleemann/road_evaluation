import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets, models

from .imputations import BaseImputer, ChannelMeanImputer
from .utils import rescale_channel

# Dataset classes with imputation.
class ImputedDataset(torch.utils.data.Dataset):
    """
    Base class for an imputed dataset according to the ROAD benchmark.
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
        transforms: transform functions for image, that are applied after imputation.
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
            imputation: BaseImputer = ChannelMeanImputer(),
            transform = None,
            target_transform = None,
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
        self.imputation = imputation # Either 'fixed' or 'linear'
        self.use_cache = use_cache
        self.cached_img = {}
        self.cached_target = {}
        self.cached_pred = {}
        self.transform = transform
        self.target_transform = target_transform

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

            # Call the imputor.
            img = self.imputation(img, bitmask)
        

            if self.use_cache: # Add to cache.
                self.cached_img[index] = img
                self.cached_target[index] = target
                self.cached_pred[index] = pred
        else:
            img = self.cached_img[index]
            target = self.cached_target[index]
            pred = self.cached_pred[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, pred

    def __len__(self):
        return len(self.base_dataset)


class ImputedDatasetMasksOnly(torch.utils.data.Dataset):
	"""
	Base class for an imputed dataset according to the ROAD benchmark.
	Unlike the dataset in the original method, the imputation is left to the 
	data_loader and this dataset only returns the masks.
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