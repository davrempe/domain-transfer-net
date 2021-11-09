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
                nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),
            
                nn.Conv2d(256, 128, kernel_size=4, stride=1, padding=0),
                nn.ReLU(inplace=True),
              	#nn.LeakyReLU(0.2, inplace=True),

                Flatten(),
                nn.Linear(128, 10)
                #nn.Conv2d(128, 10, kernel_size=1, stride=1, padding=0)
              )
		if self.use_gpu:        
			self.type(torch.cuda.FloatTensor)


	def forward(self, input):
		# TODO implement the forward pass
		return self.classify(input)

def conv_bn_lrelu(channels_in, channels_out, kernel, stride, padding, alpha, ReLU=True):
    block = nn.Sequential()
    block.add_module('conv',nn.Conv2d(channels_in, channels_out, kernel, stride, padding))
    block.add_module('batchnorm',nn.BatchNorm2d(channels_out))
    if ReLU:
        block.add_module('ReLU',nn.LeakyReLU(alpha,inplace=True))
    return block

class G(nn.Module):
	def __init__(self, channels):
		super(self.__class__,self).__init__()
		self.channels = channels
		self.block = nn.Sequential(
			# input channel will be 128
			nn.ConvTranspose2d(128, 256, kernel_size=(4,4), stride=1, padding=0),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 1, kernel_size=(4,4),stride=2,padding=1),
			nn.Tanh()            
			)
	def forward(self,input):
		output = self.block(input)
		return output

class D(nn.Module):
	def __init__(self, channels,alpha=0.2):
		super(self.__class__,self).__init__()
		self.channels = channels
		self.alpha = alpha
		self.upblock = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=(4,4), stride=2, padding=1),
			nn.LeakyReLU(self.alpha, inplace=True),
			nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1),
			nn.BatchNorm2d(128),    
			nn.LeakyReLU(self.alpha, inplace=True),
			nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1),
			nn.BatchNorm2d(256)    
        )
		self.downblock = nn.Sequential(
			nn.LeakyReLU(self.alpha,inplace=True),
			nn.Conv2d(256, 128,(4,4),1, 0),
			nn.LeakyReLU(self.alpha,inplace=True),
			nn.Conv2d(128,3,(1,1),1,0)
        )
	def forward(self, input):
		output1 = self.upblock(input)
		output = self.downblock(output1)
		return output
    
    
    
class new_G(nn.Module):
    def __init__(self, channels, use_gpu = False):
        super(self.__class__,self).__init__()
        self.channels = channels
        self.use_gpu = use_gpu
        self.block = nn.Sequential(
			# input channel will be 128
			nn.ConvTranspose2d(self.channels, 512, kernel_size=(4,4), stride=1), # output:(batch_size, 512, 4, 4)  
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=2, padding=1), # output:(batch_size, 256, 8, 8)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=2, padding=1), # output:(batch_size, 128, 16, 16)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 1, kernel_size=(4,4),stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self,input):
        output = self.block(input)
        return output

class new_D(nn.Module):
	def __init__(self, channels,alpha=0.2):
		super(self.__class__,self).__init__()
		self.channels = channels
		self.alpha = alpha
		self.block = nn.Sequential(     
			nn.Conv2d(1, 128, kernel_size=(3,3), stride=2,padding=1), # output:(batch_size, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.alpha,inplace=True),
			nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=1), # output:(batch_size, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.alpha,inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=2, padding=1), # output:(batch_size, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.alpha,inplace=True),
            nn.Conv2d(512, 3, kernel_size=(4,4), stride = 2) # output:(batch_size, 1, 1, 1)
        )

	def forward(self, input):
		output = self.block(input)
		return output

