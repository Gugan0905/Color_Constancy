''' Discriminator.py
This code file details the implementation of the Discriminator architecture and its necesssary functionalities.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import torch
import torch.nn as nn

device = torch.device('cuda') # device connected to GPU

''' Discriminator class that details the architecture of the discriminator of the conditional PatchGAN'''
class Discriminator(nn.Module):
    ''' Parameters
    input_nc - number of input channels
    ndf - number of filters
    n_layers - number of layer
    norm_layer - type of normalization
    '''
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False), #reapeating conv layers
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True) # Employs Leaky ReLU after all batch normalization
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        input = input.to(device)
        return self.model(input) #returning the model with input tied to the current device