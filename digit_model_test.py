import copy
from base_test import BaseTest
import digits_model
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms



def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))   
       
        
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
        
        MNIST_transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor()])
        
        s_train_set = torchvision.datasets.SVHN(root = './SVHN/', split='extra',download = True, transform = transforms.ToTensor())
        self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        t_train_set = torchvision.datasets.MNIST(root='./MNIST/', train=True, download = True, transform = MNIST_transform)
        self.t_train_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        s_val_set = torchvision.datasets.SVHN(root = './SVHN/', split='train',download = True, transform = transforms.ToTensor())
        self.s_val_loader = torch.utils.data.DataLoader(s_val_set, batch_size=128,
                                          shuffle=True, num_workers=8)

        s_test_set = torchvision.datasets.SVHN(root = './SVHN/', split='test', download = True, transform = transforms.ToTensor())
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
        dataiter = iter(self.s_train_loader)
        images, labels = dataiter.next()
        
        imshow(torchvision.utils.make_grid(images))
       
            
        dataiter = iter(self.t_train_loader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        
    
    def create_model(self):
        '''
        Constructs the model, converts to GPU if necessary. Saves for training.
        '''
        self.model = {}
        print('D')
        self.model['D']= digits_model.D(128, self.use_gpu)
        self.model['G'] = digits_model.G(128, self.use_gpu)
        if self.use_gpu:
            self.model['G'] = self.model['G'].cuda()    
            self.model['D'] = self.model['D'].cuda() 
        self.readClassifier('./pretrained_model/model_F_SVHN.tar')
    
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
        self.d_optimizer = optim.SGD(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg)
    
    def test_model(self):
        '''
        Tests the model and returns the loss.
        '''
        pass
    
    def create_discriminator_loss_function(self):
        '''
        Constructs the discriminator loss function.
        '''
        # s - face domain
        # t - emoji domain
        def DLoss(s_D_G,t_D_G,t_D):
            # 1 - faces through generator
            # 2 - emojis through generator
            # 3 - emojis

            label_0, label_1, label_2 = (torch.LongTensor(self.batch_size) for i in range(3))
            label_0 = Variable(label_0.cuda())
            label_1 = Variable(label_1.cuda())
            label_2 = Variable(label_2.cuda())
            label_0.data.resize_(self.batch_size).fill_(0)
            label_1.data.resize_(self.batch_size).fill_(1)
            label_2.data.resize_(self.batch_size).fill_(2)

            L_d = self.lossCE(s_D_G.squeeze(),label_0)+self.lossCE(t_D_G.squeeze(),label_1)+self.lossCE(t_D.squeeze(),label_2)
            return L_d
        
        self.d_loss_function = DLoss

    def readClassifier(self, model_name):

#         self.model['F'] = torch.load(model_name)['best_model']
#         for k, v in self.model['F'].state_dict().items():
#             print(k)
#             print('aaa')
#             print(v)
#         print(torch.load(model_name))
#         print(self.model['F'])
        old_model = torch.load(model_name)['best_model']
        old_dict = old_model.state_dict() 
        new_model = digits_model.F(3,self.use_gpu)
        new_dict = new_model.state_dict()
        new_dict = {k: v for k, v in old_dict.items() if k in new_dict}
        # 2. overwrite entries in the existing state dict
        old_dict.update(new_dict) 
        # 3. load the new state dict
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
   
    def seeResults(self,s,t):     
        if not self.use_gpu:
            s_data = Variable(s.float())
            t_data = Variable(t.float())
        else:
            s_data = Variable(s.float().cuda())
            t_data = Variable(t.float().cuda())
        
        s_F = self.model['F'](s_data)  
        s_G = self.model['G'](s_F)
        s_G = s_G.cpu()
        s_G = s_G.data
                
        to_img = torchvision.transforms.ToPILImage()
#         imshow(torchvision.utils.make_grid(t))
        npimg = torchvision.utils.make_grid(s_G).numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) 
        zero_array = np.zeros(npimg.shape)
        one_array = np.ones(npimg.shape)
        npimg = np.minimum(npimg,one_array)
        npimg = np.maximum(npimg,zero_array)
        plt.imshow(npimg)
        plt.show()

    def create_generator_loss_function(self):
        
        def GLoss(s_D_G, t_D_G, s_F, s_G_F, t, t_G, alpha, beta, gamma):
            label_0, label_1, label_2 = (torch.LongTensor(self.batch_size) for i in range(3))
            label_0 = Variable(label_0.cuda())
            label_1 = Variable(label_1.cuda())
            label_2 = Variable(label_2.cuda())
            label_0.data.resize_(self.batch_size).fill_(0)
            label_1.data.resize_(self.batch_size).fill_(1)
            label_2.data.resize_(self.batch_size).fill_(2)
           

            LGang_1 = self.lossCE(s_D_G.squeeze(),label_2)
            LGang_2 = self.lossCE(t_D_G.squeeze(),label_2)
            LGang = LGang_1 + LGang_2
            
            loss = nn.MSELoss()
            LConst = loss(s_G_F, s_F.detach())
            
            LTID = self.distance_Tdomain(t_G, t.detach())
            LTV = 0
            return LGang+alpha*LConst+beta*LTID+gamma*LTV

        self.g_loss_function = GLoss

    def create_distance_function_Tdomain(self):
        # define a distance function in T
        def Distance_T(t_1, t_2):
            distance = nn.MSELoss()
            return distance(t_1, t_2)

        self.distance_Tdomain = Distance_T

    def train_model(self, num_epochs, **kwargs):
        '''
        Trains the model.
        '''
        visualize_every_n_epoch = kwargs.get("visualize_every_n_epoch", 10)
        start_discrim_after = kwargs.get("start_discrim_train_after", 2)
        
        min_val_loss=float('inf')

        self.create_distance_function_Tdomain()

        l = min(len(self.s_train_loader),len(self.t_train_loader))-1

        g_loss = np.array([])
        d_loss = np.array([])       
        
        for epoch in range(num_epochs):
            
            train_g_loss = 0
            train_d_loss = 0
        
            if epoch < start_discrim_after:
                train_gen = True
                train_discrim = False
            else:
                if epoch % 2 == 0:
                    train_discrim = False
                    train_gen = True
                else:
                    train_discrim = True
                    train_gen = False
                    
                        
            s_data_iter = iter(self.s_train_loader)
            t_data_iter = iter(self.t_train_loader)
            
            visualize_i = np.random.randint(0,l)
            vis_s = 0
            vis_t = 0
            
#             vis_s, s_labels = s_data_iter.next()
#             vis_t, t_labels = t_data_iter.next()
                 
            for i in range(l):         
                
                if i == 0:
                    vis_s, s_labels = s_data_iter.next()
                    vis_t, t_labels = t_data_iter.next()  
                    
                    continue
                    
                s_data, s_labels = s_data_iter.next()
                t_data, t_labels = t_data_iter.next()
                
                # check terminal state in dataloader(iterator)
                if self.batch_size != s_data.size(0) or self.batch_size != t_data.size(0): continue
               
                
                self.model['G'].zero_grad()
                self.model['D'].zero_grad()
                
                t_data_3 = torch.cat((t_data, t_data, t_data), 1)

                if not self.use_gpu:
                    s_data, s_labels = Variable(s_data.float()), Variable(s_labels.long())
                    t_data, t_labels = Variable(t_data.float()), Variable(t_labels.long())
                    t_data_3 = Variable(t_data_3.float())
                else:
                    s_data, s_labels = Variable(s_data.float().cuda()), Variable(s_labels.long().cuda())
                    t_data, t_labels = Variable(t_data.float().cuda()), Variable(t_labels.long().cuda())
                    t_data_3 = Variable(t_data_3.float()).cuda()
                    
                   
                    
                t_F = self.model['F'](t_data_3)
                t_D = self.model['D'](t_data)
                s_F = self.model['F'](s_data)           
                s_G = self.model['G'](s_F)
                t_G = self.model['G'](t_F)
                
                s_G_3 = torch.cat((s_G,s_G,s_G),1)
                t_G_3 = torch.cat((t_G,t_G,t_G),1)
                s_G_F = self.model['F'](s_G_3)
                t_G_F = self.model['F'](t_G_3)
                t_D_G = self.model['D'](t_G)
                s_D_G = self.model['D'](s_G)
   
                generator_loss = self.g_loss_function(s_D_G, t_D_G, s_F, s_G_F, t_data, t_G,15,15,0)
                
                discriminator_loss = self.d_loss_function(s_D_G,t_D_G,t_D)

                if train_discrim:
                    discriminator_loss.backward()
                    self.d_optimizer.step()
                    train_d_loss += discriminator_loss.data[0]
                    
                
                if train_gen:
                    generator_loss.backward()
                    self.g_optimizer.step()  
                    train_g_loss += generator_loss.data[0]
                    
            self.seeResults(vis_s,vis_t)   
                                                                   

            if train_gen:
                train_g_loss /= l
                g_loss = np.append(g_loss,train_d_loss)
                
            
            if train_discrim:
                train_d_loss /= l
                d_loss = np.append(d_loss,train_d_loss)
                       
            print(epoch)
            print(train_g_loss)
            print(train_d_loss)
        
            
        plt.figure()
        e = np.arange(1,num_epochs+1)
        plt.plot(e,g_loss, label = 'generator loss')
        plt.plot(e,d_loss, label = 'discriminator loss')
        plt.show()
        
        
#             val_loss = self.validate(self, **kwargs)
#             print(val_loss)
            
#             self.log_losses(train_g_loss, val_loss)
#             self.log['train_d_loss'].append(train_d_loss)
            
#             if val_loss < min_val_loss:
#                 self.log_best_model()
#                 min_val_loss = val_loss

#             print('epoch:%d, train_g_loss:%4g, train_d_loss:%4g, val_loss:%4g' %(epoch,train_g_loss,train_d_loss,val_loss))


# TODO!!!
# compute the smoothness of a photo 
# not used in digit model, but used in face model
def smoothness(photo):
    pass
