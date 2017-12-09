import copy
from base_test import BaseTest
import digits_model
import numpy as np
import matplotlib.pyplot as plt
import data

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import time
import os
from data import NormalizeRangeTanh, UnNormalizeRangeTanh


class digits_model_test(BaseTest):
    '''
    Abstract class that outlines how a network test case should be defined.
    '''
    
    def __init__(self, use_gpu=True):
        super(digits_model_test, self).__init__(use_gpu)
        self.g_loss_function = None
        self.gan_loss_function = None
        self.d_loss_function = None
        self.s_val_loader = None
        self.s_test_loader = None
        self.t_test_loader = None
        self.distance_Tdomain = None
        self.s_train_loader = None
        self.t_train_loader = None
        self.batch_size = 128
        self.lossCE = nn.CrossEntropyLoss()
    
    def create_data_loaders(self):
        
        SVHN_transform = transforms.Compose([transforms.ToTensor(), NormalizeRangeTanh()])
        MNIST_transform =transforms.Compose([transforms.Scale(32),transforms.ToTensor(),NormalizeRangeTanh()])
        
        s_train_set = torchvision.datasets.SVHN(root = './SVHN/', split='extra',download = True, transform = SVHN_transform)
        self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        t_train_set = torchvision.datasets.MNIST(root='./MNIST/', train=True, download = True, transform = MNIST_transform)
        self.t_train_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        s_test_set = torchvision.datasets.SVHN(root = './SVHN/', split='test', download = True, transform = SVHN_transform)
        self.s_test_loader = torch.utils.data.DataLoader(s_test_set, batch_size=128,
                                         shuffle=False, num_workers=8)
        
        t_test_set = torchvision.datasets.MNIST(root='./MNIST/', train=False, download = True, transform = MNIST_transform)
        self.t_test_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128,
                                          shuffle=False, num_workers=8)

    def visualize_single_batch(self):
        '''
        Plots a minibatch as an example of what the data looks like.
        '''
        # get some random training images
        dataiter_s = iter(self.s_train_loader)
        images_s, labels_s= dataiter_s.next()        
        
        dataiter_t = iter(self.t_train_loader)
        images_t, labels_t = dataiter_t.next()        
        
        unnormRange = UnNormalizeRangeTanh()
        img = torchvision.utils.make_grid(unnormRange(images_s[:8]), nrow=4, padding=3)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) 
       
    def create_model(self):
        '''
        Constructs the model, converts to GPU if necessary. Saves for training.
        '''
        self.model = {}
        print('D')
        self.model['D']= digits_model.D(128)
        self.model['G'] = digits_model.G(128)
        if self.use_gpu:
            self.model['G'] = self.model['G'].cuda()    
            self.model['D'] = self.model['D'].cuda()
            
        self.readClassifier('./pretrained_model/model_F_SVHN_NormRange.tar')
        
        #Test
        model = torch.load('./pretrained_model/model_classifier_MNIST_NormRange.tar')
        self.model['MNIST_classifier'] = model['best_model']
        
    def create_loss_function(self):
        
        self.lossCE = nn.CrossEntropyLoss().cuda()        
        self.lossMSE = nn.MSELoss().cuda()
        label_0, label_1, label_2 = (torch.LongTensor(self.batch_size) for i in range(3))
        label_0 = Variable(label_0.cuda())
        label_1 = Variable(label_1.cuda())
        label_2 = Variable(label_2.cuda())
        label_0.data.resize_(self.batch_size).fill_(0)
        label_1.data.resize_(self.batch_size).fill_(1)
        label_2.data.resize_(self.batch_size).fill_(2)
        self.label_0 = label_0
        self.label_1 = label_1
        self.label_2 = label_2
        
        self.create_distance_function_Tdomain()
        self.create_discriminator_loss_function()
        self.create_generator_loss_function()

    def create_optimizer(self):
        '''
        Creates and saves the optimizer to use for training.
        '''
        g_lr = 1e-3
        g_reg = 1e-6
        self.g_optimizer = optim.Adam(self.model['G'].parameters(), lr=g_lr, weight_decay=g_reg)
        
        d_lr = 1e-3
        d_reg = 1e-6
        #self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg) #TODO: change to SGD? (according to GAN hacks)
        self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg)

    def readClassifier(self, model_name):

        old_model = torch.load(model_name)['best_model']
        old_dict = old_model.state_dict() 
        new_model = digits_model.F(3,self.use_gpu)
        new_dict = new_model.state_dict()
        new_dict = {k: v for k, v in old_dict.items() if k in new_dict}
        old_dict.update(new_dict) 
        new_model.load_state_dict(new_dict)
        self.model['F'] =new_model
        
        for param in self.model['F'].parameters():
            param.requires_grad = False
        
    def validate(self, **kwargs):
        '''
        Evaluate the model on the validation set.
        '''
        gan_loss_weight = kwargs.get("gan_loss_weight", 1e-3)
        val_loss = 0
        self.model['G'].eval()
        samples = np.random.randint(0,len(s_val_set),size = 5)
        for i in samples:
            s_data = s_val_set[i]
            s_G = self.model['G'](s_data)
            s_generator = self.model['G'](s_data)
            s_classifier = self.model['F'](s_data)
            s_G_classifer = self.model['G'](s_classifier)
            s_D_generator = self.model['D'](s_generator)

            g_loss, _, _ = self.g_loss_function(fake_curve_v, prepared_data)
            gan_loss = self.gan_loss_function(logits_fake)
            val_loss += g_loss.data[0] + gan_loss_weight * gan_loss.data[0]
        val_loss /= len(self.val_loader)
        self.model['G'].train()
        return val_loss
   
    def seeResults(self, s_data, s_G):     
        s_data = s_data.cpu().data
        s_G = s_G.cpu().data     
        # Unnormalize MNIST images
        #unnorm_SVHN = data.UnNormalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #unnorm_MNIST = data.UnNormalize((0.1307,), (0.3081,))
        unnormRange = UnNormalizeRangeTanh()
        self.imshow(torchvision.utils.make_grid(unnormRange(s_data[:16]), nrow=4))
        self.imshow(torchvision.utils.make_grid(unnormRange(s_G[:16]), nrow=4))
    
    def imshow(self, img):
        plt.figure()
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) 
        zero_array = np.zeros(npimg.shape)
        one_array = np.ones(npimg.shape)
        npimg = np.minimum(npimg,one_array)
        npimg = np.maximum(npimg,zero_array)
        plt.imshow(npimg)
        plt.show()
    
    def create_discriminator_loss_function(self):
        '''
        Constructs the discriminator loss function.
        '''
        # s - face domain
        # t - emoji domain
        def DLoss(s_D_G,t_D_G,t_D):
            
            L_d = self.lossCE(s_D_G.squeeze(), self.label_0) + self.lossCE(t_D_G.squeeze(), self.label_1) + self.lossCE(t_D.squeeze(), self.label_2)
            
            return L_d
        
        self.d_loss_function = DLoss
    def create_generator_loss_function(self):
        
        def GLoss(s_F, s_G_F, s_D_G, t, t_G, t_D_G, alpha, beta, gamma):

            LGang_1 = self.lossCE(s_D_G.squeeze(), self.label_2)
            LGang_2 = self.lossCE(t_D_G.squeeze(), self.label_2)
            LGang = LGang_1 + LGang_2
            
            LConst = self.lossMSE(s_G_F, s_F.detach())
            
            LTID = self.distance_Tdomain(t_G, t.detach())
            LTV = 0
            return LGang+alpha*LConst+beta*LTID+gamma*LTV

        self.g_loss_function = GLoss

    def create_distance_function_Tdomain(self):
        # define a distance function in T
        def Distance_T(t_1, t_2):
            distance = self.lossMSE
            return distance(t_1, t_2)

        self.distance_Tdomain = Distance_T

    def train_model(self, num_epochs, **kwargs):
        '''
        Trains the model.
        '''
        visualize_batches = kwargs.get("visualize_batches", 50)        
        save_batches = kwargs.get("save_batches", 200)        
        test_batches = kwargs.get("test_batches", 200)
        
        logdir = './log/' + str(int(time.time()))
        os.mkdir(logdir)
        self.log['logdir'] = logdir + '/'

        l = min(len(self.s_train_loader),len(self.t_train_loader))

        self.log['d_train_loss'] = []
        self.log['g_train_loss'] = []
        self.log['test_loss'] = []
        self.log['test_accuracy'] = []
        self.log['test_batches'] = test_batches

        SVHN_count = 0
        total_batches = 0
        d_runloss = 0
        g_runloss = 0
        
        for epoch in range(num_epochs):
            
            s_data_iter = iter(self.s_train_loader)
            t_data_iter = iter(self.t_train_loader)
            
            for i in range(l):         
                
                SVHN_count += 1               
                if SVHN_count >= len(self.s_train_loader):
                    SVHN_count = 0
                    s_data_iter = iter(self.s_train_loader)
                    
                s_data, s_labels = s_data_iter.next()
                t_data, t_labels = t_data_iter.next()
                
                # check terminal state in dataloader(iterator)
                if self.batch_size != s_data.size(0) or self.batch_size != t_data.size(0): continue
                total_batches += 1

               
                if not self.use_gpu:
                    s_data, s_labels = Variable(s_data.float()), Variable(s_labels.long())
                    t_data, t_labels = Variable(t_data.float()), Variable(t_labels.long())
                else:
                    s_data, s_labels = Variable(s_data.float().cuda()), Variable(s_labels.long().cuda())
                    t_data, t_labels = Variable(t_data.float().cuda()), Variable(t_labels.long().cuda())
                    
                # train discriminator
                
                for p in self.model['D'].parameters(): 
                    p.requires_grad = True 
                self.model['D'].zero_grad()
 
                s_F = self.model['F'](s_data)
                s_G = self.model['G'](s_F)
                s_G_detach = s_G.detach()
                s_D_G = self.model['D'](s_G_detach)
                
                t_data_3 = torch.cat((t_data,t_data,t_data),1)
                t_F = self.model['F'](t_data_3)
                t_G = self.model['G'](t_F)
                t_G_detach = t_G.detach()
                t_D_G = self.model['D'](t_G_detach)
                
                t_D = self.model['D'](t_data)
                
                D_loss = self.d_loss_function(s_D_G, t_D_G, t_D)
                D_loss.backward()
                self.d_optimizer.step()
                
                # train generator
                for p in self.model['D'].parameters(): 
                    p.requires_grad = False 
                self.model['G'].zero_grad()
                
                s_D_G = self.model['D'](s_G)
                s_G_3 = torch.cat((s_G,s_G,s_G),1)
                s_G_F = self.model['F'](s_G_3)
                
                t_D_G = self.model['D'](t_G)
                
                G_loss = self.g_loss_function(s_F, s_G_F, s_D_G, t_data, t_G, t_D_G,15,15,0)
                G_loss.backward()
                self.g_optimizer.step()
                
                d_runloss += D_loss.data[0]               
                g_runloss += G_loss.data[0]
               
                if total_batches % visualize_batches == 0:
                    s_F = self.model['F'](s_data)
                    s_G = self.model['G'](s_F)
                    self.seeResults(s_data, s_G)   
        
                    d_train_loss = d_runloss / visualize_batches
                    g_train_loss = g_runloss / visualize_batches
                    d_runloss = 0
                    g_runloss = 0
                    
                    self.log['d_train_loss'].append(d_train_loss)                   
                    self.log['g_train_loss'].append(g_train_loss)
                    
                    print("Epoch %d  batches %d" %(epoch, i))
                    print("d_train_loss: %f, g_train_loss %f" % (d_train_loss, g_train_loss))
                
                if total_batches % test_batches == 0:
                    accu = self.test_model()
                    self.log['test_accu'] = format(100*accu, '.3f')
                    
                if total_batches % save_batches == 0:
                    self.log['best_model'] = self.model
                    checkpoint = self.log['logdir'] + self.log['test_accu'] + '_' + str(epoch) + '_' + str(i) + '.tar'
                    torch.save(self.log, checkpoint)
                    
    
    def test_model(self):
        '''
        Tests the model and returns the loss.
        '''
        total = 0
        correct = 0
        running_loss = 0
        s_data_iter = iter(self.s_test_loader)

        for i in range(len(s_data_iter)):         

            s_data, s_labels = s_data_iter.next()
            s_labels = s_labels.numpy().squeeze()
            np.place(s_labels, s_labels == 10, 0)
            s_labels = torch.from_numpy(s_labels)


            # check terminal state in dataloader(iterator)
            if self.batch_size != s_data.size(0): continue

            if not self.use_gpu:
                s_data, s_labels = Variable(s_data.float()), Variable(s_labels.long())
            else:
                s_data, s_labels = Variable(s_data.float().cuda()), Variable(s_labels.long().cuda())

            s_F = self.model['F'](s_data)
            s_G = self.model['G'](s_F)
            
            if i == 0:
                self.seeResults(s_data, s_G)   
            
            outputs = self.model['MNIST_classifier'](s_G)
            loss = self.lossCE(outputs, s_labels)
            running_loss += loss.data[0]
            total += s_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == s_labels.data).sum()

        accuracy = 1. * correct / total
        running_loss /= len(s_data_iter)
        print('Test on MNIST classifier\n  loss: %.4f   accuracy: %.3f%%' % (running_loss, 100 * accuracy))
        self.log['test_loss'].append(running_loss)
        self.log['test_accuracy'].append(correct)
        return accuracy

# TODO!!!
# compute the smoothness of a photo 
# not used in digit model, but used in face model
def smoothness(photo):
    pass
