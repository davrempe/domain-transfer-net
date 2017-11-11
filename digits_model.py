import torch
import torch.nn as nn

import numpy as np

# TODO implement each reasonable part of the digits domain transer network as its own module class
# for example the f, g, and discriminator networks. Then we should have one class that connects them all
# Make sure to put any variables on the GPU if gpu is enabled (using Variable.cuda())

# TODO top level class that connects them all
class DigitTransferNet(nn.Module):
	'''
	Network to perform style transer between MNIST and SVHN images.
	'''
	def __init__(self, use_gpu=False):
		self.use_gpu = use_gpu
		# TODO instantiate all parts of the network
		

	def forward(self, input):
		# TODO implement the forward pass
		pass

# TODO f function (mnist classifier)
class F(nn.Module):
	'''
	MNIST digit classifier.
	'''
	def __init__(self, use_gpu=False):
		self.use_gpu = use_gpu
		# TODO instantiate all parts of the network
		# e.g. self.classify = nn.Sequential(conv layers.... + output layer)		

	def forward(self, input):
		# TODO implement the forward pass
		# e.g. call self.classify(input)
		pass


# TODO g funciton

# TODO discriminator function
