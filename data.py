import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

# TODO create a class for each dataset EXCEPT MNIST (this is already built into pytorch)
# If the dataset comes pre-split into train/test we should write a separate class for each.

# For example the street view house number might look something like this...
# The cropped version of the dataset is in a weird *.mat format, see https://stackoverflow.com/questions/29185493/read-svhn-dataset-in-python for instructions to load with numpy
class SVHNDataset(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
	
	def __len__(self):
		# TODO return length of the dataset
		pass

	def __getitem(self, idx):
		# TODO return the data point at idx
		pass
