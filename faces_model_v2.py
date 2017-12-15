import torch
import torch.nn as nn

import numpy as np

class ConvTransBNConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self.__class__,self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(self.in_c, self.out_c, kernel_size=4, stride=2, padding=1), # bias=False),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU(inplace=True),
#             nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.out_c),
            nn.ReLU(inplace=True)
#             nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, input):
        return self.block(input)

class G(nn.Module):
    '''
    Generator for face to emoji domain transfer net.
    Takes in 128d representation from OpenFace Net and outputs 64x64 RGB image.
    - in_channels : number of channels in input data
    '''
    def __init__(self, in_channels):
        super(self.__class__,self).__init__()
        self.in_channels = in_channels
        
        self.g = nn.Sequential(
            # start with (128, 1, 1), l/w double each layer
            nn.ConvTranspose2d(self.in_channels, 512, kernel_size=4, stride=1, padding=0), #, bias=False), # width 4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.out_c),
            nn.ReLU(inplace=True),
            ConvTransBNConv1(512, 512), # width 8
            ConvTransBNConv1(512, 512), # 16
            ConvTransBNConv1(512, 256), # 32
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1), #, bias=False), #64
            nn.Tanh() 
        )
        
    def forward(self,input):
        # input is 864d for openface, 512 for sphereface
        conv_in = input.view(input.size()[0], 512, 1, 1)
        output = self.g(conv_in)
        return output
    
class ConvBNLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(self.__class__,self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.alpha = alpha
        
        self.block = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.out_c),
            nn.LeakyReLU(self.alpha, inplace=True)
        )
    
    def forward(self, input):
        return self.block(input)

class D(nn.Module):
    '''
    Discriminator for face to emoji domain transfer net.
    Takes in 3x96x96 emoji image and outputs (3, 1, 1) classification.
    - channels: bases number of filters in each layer on this channel number
    '''
    def __init__(self, channels, alpha=0.2):
        super(self.__class__,self).__init__()
        self.channels = channels
        self.alpha = alpha
        
        self.block = nn.Sequential(     
            ConvBNLRelu(3, self.channels, self.alpha), # 64 channels
            ConvBNLRelu(self.channels, self.channels*2, self.alpha), # 128
            ConvBNLRelu(self.channels*2, self.channels*4, self.alpha), # 256 .... etc. 
            ConvBNLRelu(self.channels*4, self.channels*2, self.alpha), #128
            ConvBNLRelu(self.channels*2, self.channels, self.alpha), # 64
            nn.Conv2d(self.channels, 3, kernel_size=3, stride=1)
        )

    def forward(self, input):
        output = self.block(input)
        return output