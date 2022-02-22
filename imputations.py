# Implementations of our imputation models.
import torch
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class BaseImputer():

	def __call__(self, img: torch.Tensor, mask: torch.Tensor)-> torch.Tensor:
		""" Call the Imputation function to fill the masked pixels in an image.
			:param img: original image (C,H,W)-tensor
			:param mask: (H,W)-tensor with a binary mask. 0 indicates pixels absent, 1 indicates pixels present.
			:returns: a (C,H,W) tensor, where the original values are kept, if the mask for the pixels is 1 or imputed otherwise.
		"""
		raise NotImplementedError("Please implement an imputation function or use an existing imputor.")


# Some Imputors
class ChannelMeanImputer(BaseImputer):
	""" Impute by the mean value per channel of the image. """

	def __call__(self, img: torch.Tensor, mask: torch.Tensor):
		for c in range(len(img)):
			mean_c = img[c,:,:].mean()
			imgsubtensor = img[c,:,:]
			imgsubtensor[mask==0] = mean_c
		return img


class ZeroImputer(BaseImputer):
	def __call__(self, img: torch.Tensor, mask: torch.Tensor):
		return img*mask.unsqueeze(0)


# Code for infilling.
neighbors_weights = [((1,1), 1/12), ((0,1), 1/6), ((-1,1), 1/12), ((1,-1), 1/12), ((0,-1), 1/6), ((-1,-1), 1/12), ((1,0), 1/6), ((-1,0), 1/6)]

class NoisyLinearImputer(BaseImputer):
	def __init__(self, noise=0.01, weighting=neighbors_weights):
		"""	
			Noisy linear imputation.	
			noise: magnitude of noise to add (absolute, set to 0 for no noise)
			weighting: Weights of the neighboring pixels in the computation. 
			List of tuples of (offset, weight)
		"""
		self.noise = noise
		self.weighting = neighbors_weights
	
	@staticmethod 
	def add_offset_to_indices(indices, offset, mask_shape):
		""" Add the corresponding offset to the indices. Return new indices plus a valid bit-vector. """
		cord1 = indices % mask_shape[1]
		cord0 = indices // mask_shape[1]
		cord0 += offset[0]
		cord1 += offset[1]
		#print(cord1.shape, indices.shape)
		valid = ((cord0 < 0) | (cord1 < 0) | (cord0 >= mask_shape[0]) | (cord1 >= mask_shape[1]))
		return ~valid, indices+offset[0]*mask_shape[1]+offset[1]

	@staticmethod 
	def setup_sparse_system(mask, img, neighbors_weights):
		""" Vectorized version to set up the equation system.
			mask: (H, W)-tensor of missing pixels.
			Image: (H, W, C)-tensor of all values.
			Return (N,N)-System matrix, (N,C)-Right hand side for each of the C channels.
		"""
		maskflt = mask.flatten()
		imgflat = img.reshape((img.shape[0], -1))
		#print(imgflat.shape)
		indices = np.argwhere(maskflt==0).flatten() # Indices that are imputed in the flattened mask
		coords_to_vidx= np.zeros(len(maskflt), dtype=int)
		coords_to_vidx[indices] = np.arange(len(indices)) # lookup_indices =
		#print(coords_to_vidx[:10])
		#coords_to_vidx = {(idx[0].item(), idx[1].item()): i for i, idx in enumerate(indices)} # Coordinates to variable index
		numEquations = len(indices)
		A = lil_matrix((numEquations, numEquations)) # System matrix
		b = np.zeros((numEquations, img.shape[0]))
		sum_neighbors = np.ones(numEquations) # Sum of weights assigned
		#print("My indices:", indices[:10])
		#print("Num indices: ", len(indices))
		for n in neighbors_weights:
			offset, weight = n[0], n[1]
			#print("Using: ", offset, weight)
			# Sum of the neighbors.
			# Take out outliers
			valid, new_coords = NoisyLinearImputer.add_offset_to_indices(indices, offset, mask.shape)
			
			valid_coords = new_coords[valid]
			valid_ids = np.argwhere(valid==1).flatten()
			#print(valid_ids[:10], valid_coords[:10])
			#print("Valid:", valid_ids.shape)
			
			# Add values to the right hand-side
			has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
			has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
			#print(has_values_ids[:10], has_values_coords[:10])
			#print("Has Values:", has_values_coords.shape)
			b[has_values_ids, :] -= weight*imgflat[:, has_values_coords].T
			
			# Add weights to the system (left hand side)
			has_no_values = valid_coords[maskflt[valid_coords] < 0.5] # Find coordinates in the system.
			variable_ids = coords_to_vidx[has_no_values]
			has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
			
			#print("Has No Values:", has_no_values.shape)
			A[has_no_values_ids, variable_ids] = weight
			
			# Reduce weight for invalid
			#print(np.argwhere(valid==0).flatten()[:10])
			sum_neighbors[np.argwhere(valid==0).flatten()] = sum_neighbors[np.argwhere(valid==0).flatten()] - weight

		A[np.arange(numEquations),np.arange(numEquations)] = -sum_neighbors  
		return A, b

	def __call__(self, img: torch.Tensor, mask: torch.Tensor):
		""" Our linear inputation scheme. """
		"""
		This is the function to do the linear infilling 
		img: original image (C,H,W)-tensor;
		mask: mask; (H,W)-tensor

		"""
		imgflt = img.reshape(img.shape[0], -1)
		maskflt = mask.reshape(-1)
		indices_linear = np.argwhere(maskflt==0).flatten() # Indices that need to be imputed.
		# Set up sparse equation system, solve system.
		A, b = NoisyLinearImputer.setup_sparse_system(mask.numpy(), img.numpy(), neighbors_weights)
		res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

		# Fill the values with the solution of the system.
		img_infill = imgflt.clone()
		img_infill[:, indices_linear] = res.t() + self.noise*torch.randn_like(res.t())
			
		return img_infill.reshape_as(img)