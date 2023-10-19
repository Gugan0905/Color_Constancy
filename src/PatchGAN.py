''' PatchGAN.py
This code file details the implementation of the PatchGAN architecture and its necesssary functionalities.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import torch
import torch.nn as nn
from torchsummary import summary

from UnetGenerator import UnetGenerator
from Discriminator import Discriminator

device = torch.device('cuda') # current GPU device

'''Class PatchGAN that details the implementation of the PatchGAN with the combination of the Generator and Discriminators defined
in their respective class scripts'''
class PatchGAN():
        def __init__(self):
            pass #empty initialization

        '''Pix2Pix Generator and Discriminator Architecture
        custom weights initialization called on generator and discriminator
        Parameters
        net - network to be passed
        init_type - type of weight initialization
        scaling - weight scaling factor (std deviation)
        '''
        def init_weights(self, net, init_type='normal', scaling=0.02):
            def init_func(m):  # define the initialization function
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
                    torch.nn.init.normal_(m.weight.data, 0.0, scaling)
                elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                    torch.nn.init.normal_(m.weight.data, 1.0, scaling)
                    torch.nn.init.constant_(m.bias.data, 0.0)

            print('initialize network with %s' % init_type)
            net.apply(init_func)  # apply the initialization function <init_func>

        '''Function to call the generator and discriminator architectures and combine them to generate the conditional
        PatchGAN model
        returns:
        the generator and discriminator model instances initialized with the required parameters'''
        def setup_model(self):
            generator = UnetGenerator(3, 3, 64, norm_layer= nn.BatchNorm2d, use_dropout=False).to(device=device).float()
            #generator generates the fake images
            self.init_weights(generator, 'normal', scaling=0.02)
            generator = generator.to(device=device)

            print("Summary of the Generator:") #displaying the model summary
            print(summary(generator,(3,256,256)))
            print("-------------------------------------")

            discriminator = Discriminator(input_nc = 3*2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device).float()
            # discriminator takes in the ground truth as well as generated input
            self.init_weights(discriminator, 'normal', scaling=0.02)
            discriminator = discriminator.to(device = device)

            print("Summary of the Discriminator:")
            print(summary(discriminator,(6,256,256))) #displaying the model summary
            print("-------------------------------------")

            return generator, discriminator






