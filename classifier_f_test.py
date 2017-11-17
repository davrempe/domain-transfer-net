from base_test import BaseTest
from data import SVHNDataset
import digits_model

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))        
    
class classifierFTest(BaseTest):
    
    def __init__(self, use_gpu=True):
        super(self.__class__, self).__init__(use_gpu)
    
    def create_data_loaders(self):
        train_set = SVHNDataset(split='extra')
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                          shuffle=True, num_workers=8)
        test_set = SVHNDataset(split='test')
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                         shuffle=False, num_workers=8)
        
    def visualize_single_batch(self):
        # get some random training images
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()
        
        # show images
        imshow(torchvision.utils.make_grid(images))
        
    def create_model(self):
        self.model = digits_model.F(3, self.use_gpu)
    
    def create_loss_function(self):
        self.loss_function = nn.CrossEntropyLoss()
        if self.use_gpu:
            self.loss_function.type(torch.cuda.FloatTensor)

    def create_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def train_model(self, num_epochs, **kwargs):
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if not self.use_gpu:
                    inputs, labels = Variable(inputs.float()), Variable(labels.long())
                else:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                #running_loss += loss.data[0]
                #if i % 2000 == 1999:    # print every 2000 mini-batches
                #    print('[%d, %5d] loss: %.3f' %
                #          (epoch + 1, i + 1, running_loss / 2000))
                #    running_loss = 0.0
            
            print('[%dth epoch] training loss: %.3f' % (epoch + 1, loss.data[0]))

        print('Finished Training')
   
    def test_model(self):
        for data in testloader:
            inputs, labels = data
            
            if not self.use_gpu:
                inputs, labels = Variable(inputs.float()), Variable(labels.long())
            else:
                inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
            
            outputs = self.model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            loss = self.loss_function(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        
    