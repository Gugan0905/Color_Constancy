''' UnetGenerator.py
This code file details the implementation of the Unet Generator architecture and its necesssary functionalities.
The generator is termed UNet as it resembles the architecture of a Unet with serial downsampling and upsampling.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import torch
import torch.nn as nn

from UnetSkipConnectionBlock import UnetSkipConnectionBlock

device = torch.device('cuda') # current GPU device

''' Class UnetGenerator that details the entire architecture of the Generator of 
conditional GAN with the UNet Skip connection block'''
class UnetGenerator(nn.Module):
    ''' Parameters
    input_nc - number of input channels
    output_nc - number of output channels
    nf - number of filters
    norm_layer - type of normalization
    use_dropout - Flag to use dropout
    '''
    def __init__(self, input_nc, output_nc, nf = 64, norm_layer = nn.BatchNorm2d, use_dropout = False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        #print(unet_block)

        # add intermediate block with nf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        # gradually reduce the number of filters from nf * 8 to nf.
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        # add the outermost block
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        input = input.to(device)
        return self.model(input)

