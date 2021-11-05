# %% 
import os
import numpy as np 
import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision.utils import save_image

import shutil

# %%
def del_folder(path, version):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if os.path.exists(os.path.join(path, version)):
        shutil.rmtree(os.path.join(path, version))
    
def make_folder(path, version):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''    
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x

def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''    
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def str2bool(v):
    return v.lower() in ('true')

def to_LongTensor(x, *arg):
    Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    return Tensor(x, *arg)

# custom weights initialization called on netG and netD
def weights_init(m):
    '''
    custom weights initializaiont called on G and D, from the paper DCGAN

    Args:
        m (tensor): the network parameters
    '''    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_one_image(sample_path, real_images, fake_images, epoch, number=0):

    make_folder(sample_path, str(epoch) + '/real_images')
    make_folder(sample_path, str(epoch) + '/fake_images')
    real_images_path = os.path.join(sample_path, str(epoch), 'real_images')
    fake_images_path = os.path.join(sample_path, str(epoch), 'fake_images')

    # saved image must more than 10000 sheet
    # the number of the generaed images must larger than 10000
    while len(os.listdir(real_images_path)) <= 10000:
        for i in range(real_images.size(0)):
            # save real image
            one_real_image = real_images[i]
            save_image(
                one_real_image.data, 
                os.path.join(real_images_path, '{}_real.png'.format(number)),
                normalize=True
            )

            # save fake image
            one_fake_image = fake_images[i]
            save_image(
                one_fake_image.data,
                os.path.join(fake_images_path, '{}_fake.png'.format(number)),
                normalize=True
            )

            number += 1
        
        if number == 10000:
            break

def save_sample(path, images, epoch):
    '''
    save the tensor sample to nrow=10 image

    Args:
        path (str): saved path
        images (tensor): images want to save
        epoch (int): now epoch int, for the save image name
    '''    
    save_image(images.data[:100], os.path.join(path, '{}.png'.format(epoch)), normalize=True, nrow=10)

def compute_acc(real_aux, fake_aux, labels, gen_labels):
    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    return d_acc