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

    
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
        
	def forward(self, x):
		N, C, H, W = x.size() # read in N, C, H, W
		return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image    
    
class F(nn.Module):
	'''
	MNIST digit classifier.
	'''
	def __init__(self, input_channel, use_gpu=False):
		super(F, self).__init__()
		self.use_gpu = use_gpu
		self.classify = nn.Sequential(
                nn.Conv2d(input_channel, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
            
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
            
                nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0),
                nn.ReLU(inplace=True),
                
                Flatten(),
                nn.Linear(128, 10)
              )
		if self.use_gpu:        
			self.type(torch.cuda.FloatTensor)


	def forward(self, input):
		# TODO implement the forward pass
		return self.classify(input)


# TODO g funciton

# TODO discriminator function
