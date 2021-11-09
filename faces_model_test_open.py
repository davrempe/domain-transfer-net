import copy
import time
from base_test import BaseTest
import faces_model
import digits_model
import numpy as np
import matplotlib.pyplot as plt
import data
from open_face_model import OpenFace
from net_sphere import sphere20a

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms       
import torchvision.datasets as dset
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
            [data.ResizeTransform(96), data.NormalizeRangeTanh()]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        emoji_transform = transforms.Compose(
            [data.ResizeTransform(96), data.NormalizeRangeTanh()]) #transforms.Normalize((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))])
        
        #s_train_set = dset.CelebA('./datasets/', split="train", transform=msface_transform, download=False)
        #s_train_set = data.MSCeleb1MDataset('./datasets/ms-celeb-1m/data/', 'train', transform=msface_transform)
        #self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128, shuffle=True, num_workers=8)
        
        s_train_set = data.CelebADataset('./datasets/celeba/img_align_celeba/', './datasets/celeba/', 'train', transform=msface_transform)
        self.s_train_loader = torch.utils.data.DataLoader(s_train_set, batch_size=128, shuffle=True, num_workers=8)
        t_train_set = data.EmojiDataset('./datasets/emoji_data/', 0, 1000000, transform=emoji_transform)
        self.t_train_loader = torch.utils.data.DataLoader(t_train_set, batch_size=128, shuffle=True, num_workers=8)
        
        #s_test_set = data.MSCeleb1MDataset('./datasets/ms-celeb-1m/data/', 'test', transform=msface_transform)
        #self.s_test_loader = torch.utils.data.DataLoader(s_test_set, batch_size=128, shuffle=False, num_workers=8)

        s_test_set = data.CelebADataset('./datasets/celeba/img_align_celeba/', './datasets/celeba/', 'test', transform=msface_transform)
        self.s_test_loader = torch.utils.data.DataLoader(s_test_set, batch_size=128, shuffle=False, num_workers=8)

    def visualize_single_batch(self):
        '''
        Plots a minibatch as an example of what the data looks like.
        '''
        dataiter_s = iter(self.s_train_loader)
        images_s = dataiter_s.next()
        
        dataiter_t = iter(self.t_train_loader)
        images_t = dataiter_t.next()
    
        unnorm_ms = data.UnNormalizeRangeTanh() #((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        unnorm_emoji = data.UnNormalizeRangeTanh() #((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))

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
        self.model['G'] = faces_model.G(in_channels=864)
        if self.use_gpu:
            self.model['G'] = self.model['G'].cuda()    
            self.model['D'] = self.model['D'].cuda()
            
        self.prepare_openface('./pretrained_model/openface.pth', self.use_gpu)
#         self.prepare_sphereface('./pretrained_model/sphere20a_20171020.pth', self.use_gpu)
        
        self.up96 = nn.Upsample(size=(96,96), mode='bilinear')

#         self.model['F'].register_backward_hook(self.check_grad)


    def check_grad(self, module, grad_input, grad_output):
        print('in')
        print(grad_input)
        print('out')
        print(grad_output)
            
    def prepare_openface(self, pretrained_params_file, use_gpu=True):
        f_model = OpenFace(use_gpu)
        f_model.load_state_dict(torch.load(pretrained_params_file))
        
        # don't want to update params in pretrained model
        for param in f_model.parameters():
            param.requires_grad = False
        
        self.model['F'] = f_model
        
    def prepare_sphereface(self, pretrained_params_file, use_gpu=True):
        f_model = sphere20a(feature=True)
        f_model.load_state_dict(torch.load(pretrained_params_file))
        # don't want to update params in pretrained model
        for param in f_model.parameters():
            param.requires_grad = False
        
        self.model['F'] = f_model
        if use_gpu:
            self.model['F'] = sefl.model['F'].cuda()
        
        
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
        g_lr = 5e-4
        g_reg = 0#1e-6
        self.g_optimizer = optim.Adam(self.model['G'].parameters(), lr=g_lr, betas=(0.9, 0.999), weight_decay=g_reg)
        
        d_lr = 5e-4
        d_reg = 0#1e-6
        self.d_optimizer = optim.Adam(self.model['D'].parameters(), lr=d_lr, betas=(0.9, 0.999), weight_decay=d_reg)
    
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
   
    def seeResultsSrc(self, s_data, s_G):     
        s_data = s_data.cpu().data
        s_G = s_G.cpu().data
                
        # Unnormalize images
        unnorm_ms = data.UnNormalizeRangeTanh() #((0.5,0.5,0.5), (0.5,0.5,0.5))
        unnorm_emoji = data.UnNormalizeRangeTanh() #((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))
        self.imshow(unnorm_ms(s_data[:16]))
        self.imshow(unnorm_emoji(s_G[:16]))
        
    def seeResultsTgt(self, t_data, s_G):     
        t_data = t_data.cpu().data
        s_G = s_G.cpu().data
                
        # Unnormalize images
        unnorm_emoji = data.UnNormalizeRangeTanh() #((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))
        self.imshow(unnorm_emoji(t_data[:16]))
        self.imshow(unnorm_emoji(s_G[:16]))
        
    def saveResultsSrc(self, s_data, s_G, filepath, resfilepath):     
        s_data = s_data.cpu().data
        s_G = s_G.cpu().data
                
        # Unnormalize images
        unnorm_ms = data.UnNormalizeRangeTanh() #((0.5,0.5,0.5), (0.5,0.5,0.5))
        unnorm_emoji = data.UnNormalizeRangeTanh() #((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))
        self.imsave(unnorm_ms(s_data[:16]), filepath)
        self.imsave(unnorm_emoji(s_G[:16]), resfilepath)
        
    def saveResultsTgt(self, t_data, s_G, filepath, resfilepath):     
        t_data = t_data.cpu().data
        s_G = s_G.cpu().data
                
        # Unnormalize images
        unnorm_emoji = data.UnNormalizeRangeTanh() #((0.2411, 0.1801, 0.1247), (0.3312, 0.2672, 0.2127))
        self.imsave(unnorm_emoji(t_data[:16]), filepath)
        self.imsave(unnorm_emoji(s_G[:16]), resfilepath)
        
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
        
    def imsave(self, img, filepath):
        npimg = torchvision.utils.make_grid(img, nrow=4).numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) 
        zero_array = np.zeros(npimg.shape)
        one_array = np.ones(npimg.shape)
        npimg = np.minimum(npimg,one_array)
        npimg = np.maximum(npimg,zero_array)
        plt.imsave(filepath, npimg)

    def create_generator_loss_function(self):
        
        def g_train_src_loss_function(s_D_G, s_F, s_G_F, s_G):
            L_g = self.lossCE(s_D_G.squeeze(), self.label_2)
#             MSEloss = nn.MSELoss()
#             LConst = MSEloss(s_G_F, s_F.detach())
            LConst = self.calc_similarity(s_F, s_G_F)
            LTV = self.g_smoothing_function(s_G)
#             print(L_g)
#             print(LConst)
#             print(LTV)
            # Alpha/gamma params
            return L_g + LConst * 100.0 + LTV * 0.05, LConst # alpha/gamma params

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
            B, C, H, W = s_G.size()
                            
            gen_ten = s_G.contiguous().view(B, C, H, W)
            z_ijp1 = gen_ten[:, :, 1:, :-1]
            z_ijv1 = gen_ten[:, :, :-1, :-1]
            z_ip1j = gen_ten[:, :, :-1, 1:]
            z_ijv2 = gen_ten[:, :, :-1, :-1]

            diff1 = z_ijp1 - z_ijv1
            diff1 = torch.abs(diff1) 
            diff2 = z_ip1j - z_ijv2
            diff2 = torch.abs(diff2) 
            
            diff_sum = diff1 + diff2
#             print(diff_sum)
#             dist = torch.sqrt(diff_sum)
            per_chan_avg = torch.mean(diff_sum, dim=1)
            per_image_sum = torch.sum(torch.sum(per_chan_avg, dim=1), dim=1)
            loss = torch.mean(per_image_sum)
#             print(loss)
            return loss

        self.g_smoothing_function = g_train_smoothing_functions

       
    def create_distance_function_Tdomain(self):
        # define a distance function in T
        def Distance_T(t_1, t_2):
            distance = nn.MSELoss()
            return distance(t_1, t_2)

        self.distance_Tdomain = Distance_T
        
    def calc_similarity(self, s_G, s_G_F):
        # cosine sim
#         AB_sum = torch.sum(s_G * s_G_F, dim=1)
#         A_sum = torch.sqrt(torch.sum(s_G * s_G, dim=1))
#         B_sum = torch.sqrt(torch.sum(s_G_F * s_G_F, dim=1))
#         similarity = AB_sum / (A_sum * B_sum)
#         avg_sim = torch.mean(similarity)
#         return avg_sim

#         # euclidean dist
#         diff = s_G - s_G_F
#         diff_sqr = diff * diff
#         diff_sum = torch.sum(diff_sqr, dim=1)
#         dist = torch.sqrt(diff_sum)
#         avg = torch.mean(dist)
#         return avg

        # MSE
        distance = nn.MSELoss()
        return distance(s_G, s_G_F)

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

                l1_d = self.d_train_src(s_data)
                l2_d = self.d_train_trg(t_data)
                l_d = l1_d + l2_d
                self.d_train_src_runloss += l1_d.data[0]
                self.d_train_src_sum += 1
                self.d_train_trg_runloss += l2_d.data[0]
                self.d_train_trg_sum += 1 
                      
                l1_g, lconst_g = self.g_train_src(s_data)
                l2_g = self.g_train_trg(t_data)
                l_g = l1_g + l2_g                
                self.g_train_src_runloss += l1_g.data[0]
                self.lconst_src_runloss += lconst_g.data[0]
                self.g_train_src_sum += 1  
                self.g_train_trg_runloss += l2_g.data[0]
                self.g_train_trg_sum += 1
                
                if l_d.data[0] > 1.4:
                    print('train D')
                    l_d.backward()
                    self.d_optimizer.step() 
                elif l_g.data[0] > 13:
                    print('train G')
                    l_g.backward()
                    self.g_optimizer.step()
                else:
                    print('train BOTH')
                    l_d.backward()
                    self.d_optimizer.step()
                    l_g.backward()
                    self.g_optimizer.step()

                
                if i % visualize_batches == 0:
                    # TODO: do we need to set these in eval mode?
                    s_F, s_F736 = self.model['F'](s_data)
                    s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
                    # upscale
                    s_G = self.up96(s_G) 
                    self.seeResultsSrc(s_data, s_G)
                    
                    s_F, s_F736 = self.model['F'](t_data)
                    s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
                    # upscale
                    s_G = self.up96(s_G) 
                    self.seeResultsTgt(t_data, s_G)

        
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
                    self.lconst_src_runloss = 0
                        
                    import math
                    if math.isnan(g_src_loss):
                        self.model['G'].zero_grad()
                        s_F, s_F736 = self.model['F'](s_data)
                        s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
                        # upscale
                        s_G = self.up96(s_G)     
                        s_D_G = self.model['D'](s_G)
                        s_G_F, _ = self.model['F'](s_G)
                        print(s_F)
                        print(s_G)
                        print(s_D_G)
                        print(s_G_F)

                    print("Epoch %d  batches %d" %(epoch, i))
                    print("d_src_loss: %f, g_src_loss %f, lconst_src_loss %f d_trg_loss %f, g_trg_loss %f" % (d_src_loss, g_src_loss, lconst_src_loss, d_trg_loss, g_trg_loss))
                    
                if total_batches % save_batches == 0:
                    s_F, s_F736 = self.model['F'](s_data)
                    s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
                    # upscale
                    s_G = self.up96(s_G) 
                    self.saveResultsSrc(s_data, s_G, './image_log/{}_src.png'.format(i), './image_log/{}_src_res.png'.format(i))
                    
                    s_F, s_F736 = self.model['F'](t_data)
                    s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
                    # upscale
                    s_G = self.up96(s_G) 
                    self.saveResultsTgt(t_data, s_G, './image_log/{}_tgt.png'.format(i), './image_log/{}_src_tgt.png'.format(i))
                    
                    self.log['G_model'] = self.model['G']
                    self.log['D_model'] = self.model['D']
                    self.log['d_train_src_loss'] = d_train_src_loss                    
                    self.log['g_train_src_loss'] = g_train_src_loss
                    self.log['lconst_src_loss'] = lconst_train_src_loss
#                     self.log['f_train_trg_loss'] = f_train_src_loss
                    self.log['d_train_trg_loss'] = d_train_trg_loss
                    self.log['g_train_trg_loss'] = g_train_trg_loss
                    checkpoint = './log/'+ str(int(time.time())) + '_' + str(epoch) + '_' + str(i) + '.tar'
                    torch.save(self.log, checkpoint)
        

    def d_train_src(self, s_data):
        self.model['D'].zero_grad()
        s_F, s_F736 = self.model['F'](s_data)
        s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
        # upscale
        s_G = self.up96(s_G)
        s_D_G = self.model['D'](s_G)
        loss = self.d_train_src_loss_function(s_D_G)
        return loss
#         loss.backward()
#         self.d_optimizer.step()
#         self.d_train_src_runloss += loss.data[0]
#         self.d_train_src_sum += 1    

    def g_train_src(self, s_data):
        self.model['G'].zero_grad()
        s_F, s_F736 = self.model['F'](s_data)
#         print(s_F)
        s_G = self.model['G'](torch.cat((s_F, s_F736), dim=1))
        # upscale
        s_G = self.up96(s_G) 
        s_D_G = self.model['D'](s_G)
        s_G_F, _ = self.model['F'](s_G)
#         print(s_G_F)
        loss, lconst_loss = self.g_train_src_loss_function(s_D_G, s_F, s_G_F, s_G)
        return loss, lconst_loss
#         loss.backward()
#         self.g_optimizer.step()
#         self.g_train_src_runloss += loss.data[0]
#         self.lconst_src_runloss += lconst_loss.data[0]

#         self.g_train_src_sum += 1  

    def d_train_trg(self, t_data):
        self.model['D'].zero_grad()
        t_F, t_F736 = self.model['F'](t_data)
        t_D = self.model['D'](t_data)
        t_G = self.model['G'](torch.cat((t_F, t_F736), dim=1))
        # upscale
        t_G = self.up96(t_G)  
        t_D_G = self.model['D'](t_G)
        loss = self.d_train_trg_loss_function(t_D, t_D_G)
        return loss
#         loss.backward()
#         self.d_optimizer.step()
#         self.d_train_trg_runloss += loss.data[0]
#         self.d_train_trg_sum += 1  

    def g_train_trg(self, t_data):
        self.model['G'].zero_grad()
        t_F, t_F736 = self.model['F'](t_data)
        t_G = self.model['G'](torch.cat((t_F, t_F736), dim=1))
        # upscale
        t_G = self.up96(t_G)
        t_D_G = self.model['D'](t_G)
        loss = self.g_train_trg_loss_function(t_data, t_G, t_D_G)
        return loss
#         loss.backward()
#         self.g_optimizer.step()
#         self.g_train_trg_runloss += loss.data[0]
#         self.g_train_trg_sum += 1  
