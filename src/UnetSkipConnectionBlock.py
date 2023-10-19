''' UnetSkipConnectionBlock.py
This code file details the implementation of the Unet Skip Connection Block architecture and its necesssary functionalities.
The generator is termed UNet as it resembles the architecture of a Unet with serial downsampling and upsampling.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import torch
import torch.nn as nn

''' Class UnetSkipConnectionBlock to implement the skip connection blocks that occur throughout the generator 
architecture similar to a Unet structure'''
class UnetSkipConnectionBlock(nn.Module):
    ''' Parameters
    outer_nc - number of outer channels
    inner_nc - number of inner channels
    input_nc - number of input channels
    submodule=None, outermost=False, innermost=False, - Parameters to include inner and outer extra conv layers in the architecture
    norm_layer - type of normalization
    use_dropout - Flag for dropout
    '''
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                    submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True) # Relu for the downsampling half of the UNet like architecture
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True) # Relu for the upsampling half of the UNet like architecture
        upnorm = norm_layer(outer_nc)

        if outermost: # To add extra outer convolutions
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: # To add extra inner convolutions
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else: # Deafult case
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm] # Unet downsampling
            up = [uprelu, upconv, upnorm] # Unet upsampling

            if use_dropout: # Flag to check for the use of dropout of 50%
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)