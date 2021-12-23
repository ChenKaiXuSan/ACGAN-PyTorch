'''
use the pytorch-fid to calculate the FID score.
calculate with 10k images, in different 9900 epochs, separately
'''
    
import subprocess
import shlex
import pprint
import os 

from calc_fid import *

# PATH
PATH = "/home/xchen/GANs/ACGAN-PyTorch/samples/"
# FILE_NAME = "1101_mnist_10kepochs_hatano"

fid_all_list(PATH)
