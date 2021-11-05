# %%
'''
acgan structure.
the network model architecture from the paper [ACGAN](https://arxiv.org/abs/1610.09585)
'''
import torch
import torch.nn as nn

import numpy as np
from torch.nn.modules.activation import Sigmoid
# %%
class Generator(nn.Module):
    '''
    pure Generator structure

    '''    
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels = 1, n_classes=10):
        
        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.z_dim = z_dim
        self.n_classes = n_classes

        self.label_embedding = nn.Embedding(self.n_classes, self.z_dim)
        self.linear = nn.Linear(self.z_dim, 768)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.mul(label_emb, z)

        out = self.linear(gen_input)
        out = out.view(-1, 768, 1, 1)

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        
        out = self.last(out) # (*, c, 64, 64)

        return out

# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    '''
    def __init__(self, image_size = 64, conv_dim = 64, channels = 1, n_classes = 10):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.n_classes = n_classes

        # (*, c, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # (*, 64, 32, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # (*, 128, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        
        # (*, 256, 8, 8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # output layers
        # (*, 512, 8, 8)
        # dis fc
        self.last_adv = nn.Sequential(
            nn.Linear(8*8*512, 1),
            # nn.Sigmoid()
            )
        # aux classifier fc 
        self.last_aux = nn.Sequential(
            nn.Linear(8*8*512, self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        flat = out.view(input.size(0), -1)

        fc_dis = self.last_adv(flat)
        fc_aux = self.last_aux(flat)

        return fc_dis.squeeze(), fc_aux