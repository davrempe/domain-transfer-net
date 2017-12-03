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
		self.d_loss = DLoss
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
                nn.Conv2d(input_channel, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),

            
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
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

def conv_bn_lrelu(channels_in, channels_out, kernel, stride, alpha):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel, stride),
            nn.BatchNorm1d(channels_out),
            nn.LeakyReLU(alpha))

class G(nn.Module):
	def __init__(self, channels):
		super(self.__class__,self).__init__()
		self.channels = channels
		self.block = nn.Sequential(
			# input channel will be 1024
			nn.ConvTransposed2D(self.channels, self.channels/2,kernel_size=(4,4),stride=2),
			nn.BatchNorm2D(self.channels/2),
			nn.ReLU()
			)
		self.endblock = nn.Sequential(
			nn.ConvTransposed2D(self.channels, 3,kernel_size=(4,4),stride=2)
			)
	def forward(self,input):
		output1 = self.block(input)
		output2 = self.block(output1)
		output3 = self.block(output2)
		output = self.endblock(output3)
		return output

class D(nn.Module):
	def __init__(self, channels,alpha=0.2):
		super(self.__class__,self).__init__()
		self.channels = channels
		self.alpha = alpha
		self.upblock = nn.Sequential(
			nn.Conv2D(3, 32, kernel_size=(5,5),stride=2),
			nn.BatchNorm2D(32),
			nn.LeakyReLU(self.alpha,inplace=True),
			conv_bn_lrelu(32,self.channels*2,(5,5),2,self.alpha),
			conv_bn_lrelu(self.channels*2,self.channels*4,(5,5),2,self.alpha),
			conv_bn_lrelu(self.channels*4,self.channels*8,(5,5),2,self.alpha),
			)

		self.downblock = nn.Sequential(
			conv_bn_lrelu(self.channels*8,self.channels*2,(5,5),2,self.alpha),
			conv_bn_lrelu(self.channels*2,self.channels,(5,5),2,self.alpha),
			Conv2d(self.channels,1,(5,5),2,self.alpha)
			)
	def forward(self, input):
		output1 = self.upblock(input)
		output = self.downblock(output1)
		return output
