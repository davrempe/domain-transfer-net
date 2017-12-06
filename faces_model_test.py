import copy
import time
from base_test import BaseTest
import faces_model
import digits_model
import numpy as np
import matplotlib.pyplot as plt
import data
from open_face_model import OpenFace

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms       
        
class FaceTest(BaseTest):
    '''
    For training face image to emoji DTN.
    '''
    def __init__(self, use_gpu=True):
        super(FaceTest, self).__init__(use_gpu)
        self.g_loss_function = None
        self.gan_loss_function = None
        self.d_loss_function = None
        self.g_smoothing_function = None
        self.s_val_loader = None
        self.s_test_loader = None
        self.t_test_loader = None
        self.distance_Tdomain = None
        self.s_train_loader = None
        self.t_train_loader = None
        self.batch_size = 128
        self.lossCE = nn.CrossEntropyLoss()
    
    def create_data_loaders(self):
        msface_transform = transforms.Compose(
            [data.ResizeTransform(96), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        emoji_transform = transforms.Compose(
            [data.ResizeTransform(96), transforms.Normalize((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))])
        
        s_train_set = data.MSCeleb1MDataset('./datasets/ms-celeb-1m/data/', 'train', transform=msface_transform)
        self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128, shuffle=True, num_workers=8)
        
        t_train_set = data.EmojiDataset('./datasets/emoji_data/', 0, 1000000, transform=emoji_transform)
        self.t_train_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128, shuffle=True, num_workers=8)
        
        s_test_set = data.MSCeleb1MDataset('./datasets/ms-celeb-1m/data/', 'test', transform=msface_transform)
        self.s_test_loader = torch.utils.data.DataLoader(s_test_set, batch_size=128, shuffle=False, num_workers=8)

    def visualize_single_batch(self):
        '''
        Plots a minibatch as an example of what the data looks like.
        '''
        dataiter_s = iter(self.s_train_loader)
        images_s = dataiter_s.next()
        
        dataiter_t = iter(self.t_train_loader)
        images_t = dataiter_t.next()
    
        unnorm_ms = data.UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        unnorm_emoji = data.UnNormalize((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))

        img_ms = torchvision.utils.make_grid(unnorm_ms(images_s[:16]), nrow=4)
        img_emoji = torchvision.utils.make_grid(unnorm_emoji(images_t[:16]), nrow=4)

        npimg_ms = img_ms.numpy()
        npimg_emoji = img_emoji.numpy()
        zero_array = np.zeros(npimg_ms.shape)
        one_array = np.ones(npimg_ms.shape)
        
        npimg_ms = np.minimum(npimg_ms,one_array)
        npimg_ms = np.maximum(npimg_ms,zero_array)
        npimg_emoji = np.minimum(npimg_emoji,one_array)
        npimg_emoji = np.maximum(npimg_emoji,zero_array)
        
        plt.imshow(np.transpose(npimg_ms, (1, 2, 0))) 
        plt.show()
        plt.imshow(np.transpose(npimg_emoji, (1, 2, 0))) 
        plt.show()
       
    def create_model(self):
        '''
        Constructs the model, converts to GPU if necessary. Saves for training.
        '''
        self.model = {}
        self.model['D']= faces_model.D(128, alpha=0.2)
        self.model['G'] = faces_model.G(in_channels=128)
        if self.use_gpu:
            self.model['G'] = self.model['G'].cuda()    
            self.model['D'] = self.model['D'].cuda()
            
        self.prepare_openface('./pretrained_model/openface.pth', self.use_gpu)
        
        self.up96 = nn.Upsample(size=(96,96), mode='bilinear')
        
    def prepare_openface(self, pretrained_params_file, use_gpu=True):
        f_model = OpenFace(use_gpu)
        f_model.load_state_dict(torch.load(pretrained_params_file))
        
        # don't want to update params in pretrained model
        for param in f_model.parameters():
            param.requires_grad = False
        
        self.model['F'] = f_model
        
    def create_loss_function(self):       
        
        self.lossCE = nn.CrossEntropyLoss()
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
        self.create_smoothing_loss_function()
#         self.create_encoder_loss_function()

    def create_optimizer(self):
        '''
        Creates and saves the optimizer to use for training.
        '''
        g_lr = 1e-3
        g_reg = 0 #1e-6
        self.g_optimizer = optim.Adam(self.model['G'].parameters(), lr=g_lr, weight_decay=g_reg)
        
        d_lr = 1e-3
        d_reg = 0 #1e-6
        #self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg) #TODO: change to SGD? (according to GAN hacks)
        self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, weight_decay=d_reg)

#         f_lr = 3e-3
#         f_reg = 1e-6
#         self.f_optimizer = optim.Adam(self.model['F'].parameters(), lr=f_lr, weight_decay=f_reg)
    
    def test_model(self):
        '''
        Tests the model and returns the loss.
        '''
        pass
        
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
                
        # Unnormalize images
        unnorm_ms = data.UnNormalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        unnorm_emoji = data.UnNormalize((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))
        self.imshow(unnorm_ms(s_data[:16]))
        self.imshow(unnorm_emoji(s_G[:16]))
        
    def imshow(self, img):
        plt.figure()
        npimg = torchvision.utils.make_grid(img, nrow=4).numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) 
        zero_array = np.zeros(npimg.shape)
        one_array = np.ones(npimg.shape)
        npimg = np.minimum(npimg,one_array)
        npimg = np.maximum(npimg,zero_array)
        plt.imshow(npimg)
        plt.show()

#     def create_encoder_loss_function(self):

#         def f_train_src_loss_function(s_F, s_G_F):
            
#             MSEloss = nn.MSELoss()
#             LConst = MSEloss(s_G_F, s_F.detach())
#             return LConst * 100.0 # Alpha param

#         self.f_train_src_loss_function = f_train_src_loss_function    

    def create_generator_loss_function(self):
        
        def g_train_src_loss_function(s_D_G, s_F, s_G_F, s_G):
            L_g = self.lossCE(s_D_G.squeeze(), self.label_2)
            MSEloss = nn.MSELoss()
            LConst = MSEloss(s_G_F, s_F.detach())
            LTV = self.g_smoothing_function(s_G)
#             print(L_g)
#             print(LTV)
            # Alpha/gamma params
            return L_g + LConst * 100.0 + LTV * 0.05, LConst

        self.g_train_src_loss_function = g_train_src_loss_function

        def g_train_trg_loss_function(t, t_G, t_D_G):
           
            L_g = self.lossCE(t_D_G.squeeze(), self.label_2)
            LTID = self.distance_Tdomain(t_G, t.detach())
            return L_g + LTID * 1.0 # Beta param

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
        
    def create_smoothing_loss_function(self):
        '''
        Constructs the total variation loss function.
        '''
        def g_train_smoothing_functions(s_G):
            z_ijp1 = s_G[:, :, :, 1:96]
            z_ijv1 = s_G[:, :, :, 0:95]
            z_ip1j = s_G[:, :, 1:96, :]
            z_ijv2 = s_G[:, :, 0:95, :]

            diff1 = z_ijp1 - z_ijv1
            diff1 = diff1*diff1
            diff2 = z_ip1j - z_ijv2
            diff2 = diff2*diff2
            
            diff1 = diff1[:, :, 0:95, 0:95]
            diff2 = diff2[:, :, 0:95, 0:95]

            diff_sum = diff1 + diff2
            dist = torch.sqrt(diff_sum)
            per_chan_avg = torch.mean(dist, dim=1)
            per_image_sum = torch.sum(torch.sum(per_chan_avg, dim=1), dim=1)
            loss = torch.mean(per_image_sum)
            return loss

        self.g_smoothing_function = g_train_smoothing_functions

       
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
        visualize_batches = kwargs.get("visualize_batches", 50)        
        save_batches = kwargs.get("save_batches", 200)

        min_val_loss=float('inf')

        l = min(len(self.s_train_loader),len(self.t_train_loader))

        d_train_src_loss = []
        g_train_src_loss = []
        lconst_train_src_loss = []
        d_train_trg_loss = []
        g_train_trg_loss = []
        msimg_count = 0
#         F_interval = 15
        total_batches = 0
        
        for epoch in range(num_epochs):
            
            self.d_train_src_sum = 0
            self.g_train_src_sum = 0
            self.f_train_src_sum = 0
            self.d_train_trg_sum = 0
            self.g_train_trg_sum = 0
            self.d_train_src_runloss = 0
            self.g_train_src_runloss = 0
            self.f_train_src_runloss = 0
            self.d_train_trg_runloss = 0
            self.g_train_trg_runloss = 0
            self.lconst_src_runloss = 0
           
            s_data_iter = iter(self.s_train_loader)
            t_data_iter = iter(self.t_train_loader)
            
            for i in range(l):         
                
                msimg_count += 1               
                if msimg_count >= len(self.s_train_loader):
                    msimg_count = 0
                    s_data_iter = iter(self.s_train_loader)
                    
                s_data = s_data_iter.next()
                t_data = t_data_iter.next()
                
                # check terminal state in dataloader(iterator)
                if self.batch_size != s_data.size(0) or self.batch_size != t_data.size(0): continue
                total_batches += 1
               
                if not self.use_gpu:
                    s_data = Variable(s_data.float())
                    t_data = Variable(t_data.float())
                else:
                    s_data = Variable(s_data.float().cuda())
                    t_data = Variable(t_data.float().cuda())
                
                # train by feeding ms face images 
#                 if total_batches > 1600:
#                     F_interval = 30
#                 if total_batches % F_interval == 0:
#                     self.f_train_src(s_data)

                self.d_train_src(s_data)
                self.g_train_src(s_data)
#                 self.g_train_src(s_data)
#                 self.g_train_src(s_data)
#                 self.g_train_src(s_data)
#                 self.g_train_src(s_data)
#                 self.g_train_src(s_data)

                #train by feeding emoji image
                self.d_train_trg(t_data)
#                 self.d_train_trg(t_data)
                self.g_train_trg(t_data)
#                 self.g_train_trg(t_data)
#                 self.g_train_trg(t_data)
#                 self.g_train_trg(t_data)
                
                if i % visualize_batches == 0:
                    # TODO: do we need to set these in eval mode?
                    s_F, _ = self.model['F'](s_data)
                    s_G = self.model['G'](s_F)
                    # upscale
                    s_G = self.up96(s_G) 
                    self.seeResults(s_data, s_G)  
                    
                    s_F, _ = self.model['F'](t_data)
                    s_G = self.model['G'](s_F)
                    # upscale
                    s_G = self.up96(s_G) 
                    self.seeResults(t_data, s_G)
        
                    d_src_loss = self.d_train_src_runloss / self.d_train_src_sum
                    g_src_loss = self.g_train_src_runloss / self.g_train_src_sum
                    lconst_src_loss = self.lconst_src_runloss / self.g_train_src_sum
#                     if self.f_train_src_sum != 0:
#                         f_src_loss = self.f_train_src_runloss / self.f_train_src_sum
#                     else:
#                         f_src_loss = 0
                    d_trg_loss = self.d_train_trg_runloss / self.d_train_trg_sum
                    g_trg_loss = self.g_train_trg_runloss / self.g_train_trg_sum
                    d_train_src_loss.append(d_src_loss)                    
                    g_train_src_loss.append(g_src_loss)
#                     f_train_src_loss.append(f_src_loss)
                    lconst_train_src_loss.append(lconst_src_loss)
                    d_train_trg_loss.append(d_trg_loss)                    
                    g_train_trg_loss.append(g_trg_loss)
                    self.d_train_src_sum = 0
                    self.g_train_src_sum = 0
                    self.d_train_trg_sum = 0
                    self.g_train_trg_sum = 0
                    self.d_train_src_runloss = 0
                    self.g_train_src_runloss = 0
                    self.d_train_trg_runloss = 0
                    self.g_train_trg_runloss = 0

                    print("Epoch %d  batches %d" %(epoch, i))
                    print("d_src_loss: %f, g_src_loss %f, lconst_src_loss %f d_trg_loss %f, g_trg_loss %f" % (d_src_loss, g_src_loss, lconst_src_loss, d_trg_loss, g_trg_loss))
                    
                if total_batches % save_batches == 0:
                    self.log['model'] = self.model
                    self.log['d_train_src_loss'] = d_train_src_loss                    
                    self.log['g_train_src_loss'] = g_train_src_loss
                    self.log['lconst_src_loss'] = lconst_train_src_loss
#                     self.log['f_train_trg_loss'] = f_train_src_loss
                    self.log['d_train_trg_loss'] = d_train_trg_loss
                    self.log['g_train_trg_loss'] = g_train_trg_loss
#                     checkpoint = './log/'+ str(int(time.time())) + '_' + str(epoch) + '_' + str(i) + '.tar'
#                     torch.save(self.log, checkpoint)

#             val_loss = self.validate(self, **kwargs)
#             print(val_loss)
            
#             self.log_losses(train_g_loss, val_loss)
#             self.log['train_d_loss'].append(train_d_loss)
            
#             if val_loss < min_val_loss:
#                 self.log_best_model()
#                 min_val_loss = val_loss

#             print('epoch:%d, train_g_loss:%4g, train_d_loss:%4g, val_loss:%4g' %(epoch,train_g_loss,train_d_loss,val_loss))
        

    def d_train_src(self, s_data):
        self.model['D'].zero_grad()
        #self.model['G'].zero_grad()
        # for param in self.model['D'].parameters():
        #     param.requires_grad = True
        # for param in self.model['F'].parameters():
        #     param.requires_grad = False
        # for param in self.model['G'].parameters():
        #     param.requires_grad = True
        s_F, _ = self.model['F'](s_data)
        s_G = self.model['G'](s_F)
        # upscale
        s_G = self.up96(s_G)
        s_D_G = self.model['D'](s_G)
        loss = self.d_train_src_loss_function(s_D_G)
        loss.backward()
        self.d_optimizer.step()
        self.d_train_src_runloss += loss.data[0]
        self.d_train_src_sum += 1    

    def g_train_src(self, s_data):
        self.model['G'].zero_grad()
        s_F, _ = self.model['F'](s_data)
        s_G = self.model['G'](s_F)
        # upscale
        s_G = self.up96(s_G)     
        s_D_G = self.model['D'](s_G)
        s_G_F, _ = self.model['F'](s_G)
        loss, lconst_loss = self.g_train_src_loss_function(s_D_G, s_F, s_G_F, s_G)
        loss.backward()
        self.g_optimizer.step()
        self.g_train_src_runloss += loss.data[0]
        self.lconst_src_runloss += lconst_loss.data[0]
        
        self.g_train_src_sum += 1  

#     def f_train_src(self, s_data):
#         self.model['F'].zero_grad()
#         s_F = self.model['F'](s_data)
#         s_G = self.model['G'](s_F)
#         s_G_3 = torch.cat((s_G,s_G,s_G),1)
#         s_G_F = self.model['F'](s_G_3)
#         loss = self.f_train_src_loss_function(s_F, s_G_F)
#         loss.backward()
#         self.f_optimizer.step()
#         self.f_train_src_runloss += loss.data[0]
#         self.f_train_src_sum += 1  

    def d_train_trg(self, t_data):
        self.model['D'].zero_grad()
        t_F, _ = self.model['F'](t_data)
        t_D = self.model['D'](t_data)
        t_G = self.model['G'](t_F)
        # upscale
        t_G = self.up96(t_G)  
        t_D_G = self.model['D'](t_G)
        loss = self.d_train_trg_loss_function(t_D, t_D_G)
        loss.backward()
        self.d_optimizer.step()
        self.d_train_trg_runloss += loss.data[0]
        self.d_train_trg_sum += 1  

    def g_train_trg(self, t_data):
        self.model['G'].zero_grad()
        t_F, _ = self.model['F'](t_data)
        t_G = self.model['G'](t_F)
        # upscale
        t_G = self.up96(t_G)
        t_D_G = self.model['D'](t_G)
        loss = self.g_train_trg_loss_function(t_data, t_G, t_D_G)
        loss.backward()
        self.g_optimizer.step()
        self.g_train_trg_runloss += loss.data[0]
        self.g_train_trg_sum += 1  
        
