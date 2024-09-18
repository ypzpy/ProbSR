import sys
import os
import logging
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import h5py
from scipy.linalg import sqrtm


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)
        
        
def makedir(dir_name):
    if os.path.exists(os.path.join(dir_name, 'ckpt')):
        print('Output directory already exists')    
    else:
        os.makedirs(os.path.join(dir_name, 'ckpt'))
        os.makedirs(os.path.join(dir_name, 'chains'))
        
        
def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    # file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def save_model(dir_name, epoch, model_name, lr_schedule, G, optG):
    save_dict = {
            'epoch': epoch,
            'lr_schedule': lr_schedule.state_dict(),
            'netG': G.state_dict(),
            'optG': optG.state_dict()
        }
    torch.save(save_dict, f'{dir_name}/ckpt/{model_name}.pth')
    
    
class DataFromH5File(data.Dataset):
    def __init__(self, file_name):
        data = h5py.File(file_name, 'r')
        self.hr = data['high_res']
        self.residual = data['residual']
        
    def __getitem__(self, idx):
        hr = torch.from_numpy(self.hr[idx]).float()
        residual = torch.from_numpy(self.residual[idx]).float()
        
        return residual, hr
    
    def __len__(self):
        assert self.hr.shape[0] == self.residual.shape[0], "Wrong data length"
        return self.hr.shape[0]


class DataFromH5File2(data.Dataset):
    def __init__(self, file_name):
        data = h5py.File(file_name, 'r')
        self.hr = data['high_res']
        self.lr = data['low_res']
        
    def __getitem__(self, idx):
        hr = torch.from_numpy(self.hr[idx]).float()
        lr = torch.from_numpy(self.lr[idx]).float()
        
        return lr, hr
    
    def __len__(self):
        assert self.hr.shape[0] == self.lr.shape[0], "Wrong data length"
        return self.hr.shape[0]
    

class DataFromH5File3(data.Dataset):
    def __init__(self, file_name, N_low, N_high, scale):
        data = h5py.File(file_name, 'r')
        self.N_high = N_high
        self.N_low = N_low
        
        # Code downscaling matrix
        self.H = np.zeros((N_low*N_low, N_high*N_high))

        submatrix = np.zeros((N_low,N_high))
        for i in range(N_low):
            submatrix[i,scale*i] = 1
            
        for j in range(N_low):
            self.H[N_low*j:N_low*(j+1),N_high*scale*j:N_high*(scale*j+1)] = submatrix
            
        self.hr = data['high_res']
        self.residual = data['residual']
        self.lr = data['low_res']
        
    def __getitem__(self, idx):
        
        observations = (self.H @ self.hr[idx].reshape(self.N_high**2,1)).reshape(self.N_low,self.N_low)
        observation = torch.from_numpy(observations).float()
        residual = torch.from_numpy(self.residual[idx]).float()
        low_res = torch.from_numpy(self.lr[idx]).float()
        
        return residual, observation, low_res
    
    def __len__(self):
        assert self.hr.shape[0] == self.residual.shape[0], "Wrong data length"
        return self.hr.shape[0]
    
    
class DataFromH5File4(data.Dataset):
    def __init__(self, file_name):
        data = h5py.File(file_name, 'r')
        self.lr = data['low_res']
        self.forcing = data['forcing']
        
    def __getitem__(self, idx):
        lr = self.lr[idx]
        forcing = self.forcing[idx]
        
        return forcing, lr
    
    def __len__(self):
        assert self.lr.shape[0] == self.forcing.shape[0], "Wrong data length"
        return self.lr.shape[0]
    
    
class DataFromH5File5(data.Dataset):
    def __init__(self, file_name, N_low, N_high, scale):
        data = h5py.File(file_name, 'r')

        self.hr = data['high_res']
        self.residual = data['residual']
        self.lr = data['low_res']
        
    def __getitem__(self, idx):
        
        high_res = torch.from_numpy(self.hr[idx]).float()
        residual = torch.from_numpy(self.residual[idx]).float()
        low_res = torch.from_numpy(self.lr[idx]).float()
        
        return residual, high_res, low_res
    
    def __len__(self):
        assert self.hr.shape[0] == self.residual.shape[0], "Wrong data length"
        return self.hr.shape[0]