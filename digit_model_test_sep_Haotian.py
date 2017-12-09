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
#import torch.optim.lr_scheduler.MultiStepLR as MultiStepLR 

  
               
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
    
    def create_data_loaders(self):
        #SVHN_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        #MNIST_transform =transforms.Compose([transforms.Scale(32),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

        SVHN_transform = transforms.Compose([transforms.ToTensor(), NormalizeRangeTanh()])
        MNIST_transform =transforms.Compose([transforms.Scale(32),transforms.ToTensor(),NormalizeRangeTanh()])

        
        s_train_set = torchvision.datasets.SVHN(root = './SVHN/', split='extra',download = True, transform = SVHN_transform)
        self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        t_train_set = torchvision.datasets.MNIST(root='./MNIST/', train=True, download = True, transform = MNIST_transform)
        self.t_train_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

#         s_val_set = torchvision.datasets.SVHN(root = './SVHN/', split='train',download = True, transform = SVHN_transform)
#         self.s_val_loader = torch.utils.data.DataLoader(s_val_set, batch_size=128,
#                                           shuffle=True, num_workers=8)

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

    def readClassifier(self, model_name):

        old_model = torch.load(model_name)['best_model']
        #self.model['FC'] = old_model
        old_dict = old_model.state_dict() 
        new_model = digits_model.F(3,self.use_gpu)
        new_dict = new_model.state_dict()
        new_dict = {k: v for k, v in old_dict.items() if k in new_dict}
        old_dict.update(new_dict) 
        new_model.load_state_dict(new_dict)
        self.model['F'] =new_model
        
        for param in self.model['F'].parameters():
            param.requires_grad = False    
        
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
        #self.g_scheduler = MultiStepLR(self.g_optimizer, milestones=[5, 15, 30], gamma=0.1)
        

        d_lr = 1e-3
        d_reg = 1e-6
        #self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg) #TODO: change to SGD? (according to GAN hacks)
        self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg)
        #self.d_scheduler = MultiStepLR(self.d_optimizer, milestones=[5, 15, 30], gamma=0.1)

        #f_lr = 1e-3
        #f_reg = 1e-6
        #self.f_optimizer = optim.Adam(self.model['F'].parameters(), lr=f_lr, weight_decay=f_reg)
        
#         lambda1 = lambda lr: lr*0.9
        
#         self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda = lambda1)
#         self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda = lambda1)
        
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

    def create_generator_loss_function(self):
        
        def g_train_src_loss_function(s_D_G, s_F, s_G_F):
            L_g = self.lossCE(s_D_G.squeeze(), self.label_2)
            LConst = self.lossMSE(s_G_F, s_F.detach())
            return L_g

        self.g_train_src_loss_function = g_train_src_loss_function
        
        def LConst_train_src_loss_function(s_F, s_G_F):
            
            LConst = self.lossMSE(s_G_F, s_F.detach())
            return LConst * 15
        
        self.LConst_train_src_loss_function = LConst_train_src_loss_function 
        
        def LConst_train_src_loss_function2(s_G_F, s_labels):
            
            LConst = self.lossCE(s_G_F, s_labels)
            return LConst * 15
        
        self.LConst_train_src_loss_function2 = LConst_train_src_loss_function2  

        def g_train_trg_loss_function(t, t_G, t_D_G):
           
            L_g = self.lossCE(t_D_G.squeeze(), self.label_2)
            LTID = self.distance_Tdomain(t_G, t.detach())
            return L_g + LTID * 15

        self.g_train_trg_loss_function = g_train_trg_loss_function

    def create_discriminator_loss_function(self):
        '''
        Constructs the discriminator loss function.
        '''
        # s - face domain
        # t - emoji domain
        def d_train_src_loss_function(s_D_G):

            L_d = self.lossCE(s_D_G.squeeze(), self.label_0)
            return L_d
        
        self.d_train_src_loss_function = d_train_src_loss_function

        def d_train_trg_loss_function(t_D, t_D_G):

            L_d = self.lossCE(t_D_G.squeeze(), self.label_1)+self.lossCE(t_D.squeeze(), self.label_2)
            return L_d
        
        self.d_train_trg_loss_function = d_train_trg_loss_function

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

        self.log['d_train_src_loss'] = []
        self.log['g_train_src_loss'] = []
        self.log['LConst_train_src_loss'] = []
        self.log['d_train_trg_loss'] = []
        self.log['g_train_trg_loss'] = []        
        self.log['train_batches'] = []        
        
        self.log['test_loss'] = []
        self.log['test_accuracy'] = []
        self.log['test_batches'] = []

        SVHN_count = 0
        total_batches = 0
        LConst_interval = 9999999999
        
        self.d_train_src_sum = 0
        self.g_train_src_sum = 0
        self.d_train_trg_sum = 0
        self.g_train_trg_sum = 0
        self.d_train_src_runloss = 0
        self.g_train_src_runloss = 0
        self.d_train_trg_runloss = 0
        self.g_train_trg_runloss = 0
        self.LConst_train_src_sum = 0
        self.LConst_train_src_runloss = 0
        
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
                
                s_labels = s_labels.numpy().squeeze()
                np.place(s_labels, s_labels == 10, 0)
                s_labels = torch.from_numpy(s_labels)
                
                # check terminal state in dataloader(iterator)
                if self.batch_size != s_data.size(0) or self.batch_size != t_data.size(0): continue
                total_batches += 1
               
                if not self.use_gpu:
                    s_data = Variable(s_data.float())
                    t_data = Variable(t_data.float())
                else:
                    s_data = Variable(s_data.float().cuda())
                    s_labels = Variable(s_labels.long().cuda())
                    t_data = Variable(t_data.float().cuda())
                
                if total_batches > 800:
                    LConst_interval = 1
                if total_batches > 1600:
                    LConst_interval = 999999999
                #if total_batches > 1600:
                #    LConst_interval = 9999999999
                #if total_batches % LConst_interval == 0:
                #    self.LConst_train_src(s_data)
                
                # train by feeding SVHN 
                self.d_train_src(s_data)
                for j in range(6):#6
                    self.g_train_src(s_data)
                   
        
                #train by feeding MNIST
                self.d_train_trg(t_data)                
                self.d_train_trg(t_data)
                for j in range(4):#4
                    self.g_train_trg(t_data)
                
                if total_batches % visualize_batches == 0:
                    s_F = self.model['F'](s_data)
                    s_G = self.model['G'](s_F)
                    self.seeResults(s_data, s_G)   
        
                    if self.d_train_src_sum != 0:
                        d_src_loss = self.d_train_src_runloss / self.d_train_src_sum
                    else:
                        d_src_loss = 0
                    g_src_loss = self.g_train_src_runloss / self.g_train_src_sum
                    if self.LConst_train_src_sum != 0:
                        LConst_src_loss = self.LConst_train_src_runloss / self.LConst_train_src_sum
                    else:
                        LConst_src_loss = 0
                    if self.d_train_trg_sum != 0:
                        d_trg_loss = self.d_train_trg_runloss / self.d_train_trg_sum
                    else:
                        d_trg_loss = 0
                    g_trg_loss = self.g_train_trg_runloss / self.g_train_trg_sum
                    self.log['d_train_src_loss'].append(d_src_loss)                   
                    self.log['g_train_src_loss'].append(g_src_loss)
                    self.log['LConst_train_src_loss'].append(LConst_src_loss)
                    self.log['d_train_trg_loss'].append(d_trg_loss) 
                    self.log['g_train_trg_loss'].append(g_trg_loss)
                    self.log['train_batches'].append(total_batches)
                    self.d_train_src_sum = 0
                    self.g_train_src_sum = 0
                    self.d_train_trg_sum = 0
                    self.g_train_trg_sum = 0                    
                    self.LConst_train_src_sum = 0
                    self.d_train_src_runloss = 0
                    self.g_train_src_runloss = 0
                    self.d_train_trg_runloss = 0
                    self.g_train_trg_runloss = 0                    
                    self.LConst_train_src_runloss = 0


                    print("Epoch %d  batches %d" %(epoch, i))
                    print("d_src_loss: %f, g_src_loss %f, LConst_src_loss %f d_trg_loss %f, g_trg_loss %f" % (d_src_loss, g_src_loss, LConst_src_loss, d_trg_loss, g_trg_loss))
                
                if total_batches % test_batches == 0:
                    self.test_model()
                    self.log['test_batches'].append(total_batches)

                    
                if total_batches % save_batches == 0:
                    self.log['best_model'] = self.model
                    accu = format(self.log['test_accuracy'][-1]*100, '.3f')
                    checkpoint = self.log['logdir'] + format(epoch, '02d') + '_' + format(i, '03d') + '_' + accu + '.tar'
                    torch.save(self.log, checkpoint)
                
    def d_train_src(self, s_data):
        self.model['D'].zero_grad()
        s_F = self.model['F'](s_data)
        s_G = self.model['G'](s_F)
        s_G_detach = s_G.detach()
        s_D_G = self.model['D'](s_G_detach)
        #sDG = s_D_G.cpu().data.squeeze()
        #print('g(f(s))\n', sDG[:10,:])
        loss = self.lossCE(s_D_G.squeeze(), self.label_0)
        #print('d_train_src', loss.data[0]) 
        #if loss.data[0] < 0.3:
        #    return
        loss.backward()
        self.d_optimizer.step()
        self.d_train_src_runloss += loss.data[0]
        self.d_train_src_sum += 1

    def g_train_src(self, s_data):
        self.model['G'].zero_grad()
        s_F = self.model['F'](s_data)
        s_G = self.model['G'](s_F)
        s_D_G = self.model['D'](s_G)
        
        s_G_3 = torch.cat((s_G,s_G,s_G),1)
        s_G_F = self.model['F'](s_G_3)
        loss = self.lossCE(s_D_G.squeeze(), self.label_2)

        loss.backward()
        self.g_optimizer.step()
        self.g_train_src_runloss += loss.data[0]
        self.g_train_src_sum += 1  

    def LConst_train_src(self, s_data, s_labels=None):
        self.model['G'].zero_grad()
        s_F = self.model['F'](s_data)
        s_G = self.model['G'](s_F)
        s_G_3 = torch.cat((s_G,s_G,s_G),1)
        if s_labels is None:
            s_G_F = self.model['F'](s_G_3)
            loss = self.lossMSE(s_G_F, s_F.detach()) * 15
        else:
            s_G_F = self.model['MNIST_classifier'](s_G)
            loss = LConst = self.lossCE(s_G_F, s_labels) * 15
            
        loss.backward()
        self.g_optimizer.step()
        self.LConst_train_src_runloss += loss.data[0]
        self.LConst_train_src_sum += 1  

    def d_train_trg(self, t_data):
        self.model['D'].zero_grad()
        t_data_3 = torch.cat((t_data,t_data,t_data),1) 
        t_F = self.model['F'](t_data_3)
        t_D = self.model['D'](t_data)
        t_G = self.model['G'](t_F)
        t_G_detach = t_G.detach()
        t_D_G = self.model['D'](t_G_detach)
        
        loss = self.lossCE(t_D_G.squeeze(), self.label_1)+self.lossCE(t_D.squeeze(), self.label_2)
        #print('d_train_trg', loss.data[0])
        #if loss.data[0] < 0.3:
        #    return
        loss.backward()
        self.d_optimizer.step()
        self.d_train_trg_runloss += loss.data[0]
        self.d_train_trg_sum += 1

    def g_train_trg(self, t_data):
        self.model['G'].zero_grad()
        t_data_3 = torch.cat((t_data,t_data,t_data),1) 
        t_F = self.model['F'](t_data_3)
        t_G = self.model['G'](t_F)
        t_D_G = self.model['D'](t_G)
        
        L_g = self.lossCE(t_D_G.squeeze(), self.label_2)
        LTID = self.distance_Tdomain(t_G, t_data.detach())
        loss = L_g + LTID * 15
        
        loss.backward()
        self.g_optimizer.step()
        self.g_train_trg_runloss += loss.data[0]
        self.g_train_trg_sum += 1  
    
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
        self.log['test_accuracy'].append(accuracy)
    
    def plot_accuracy(self):
        accu = self.log['test_accuracy']
        batches = self.log['test_batches']
        plt.plot(batches, accu, label='test_accuracy')
        plt.xlabel('batches')        
        plt.ylabel('test_accuracy')
        plt.legend()
        plt.show()


# TODO!!!
# compute the smoothness of a photo 
# not used in digit model, but used in face model
def smoothness(photo):
    pass
