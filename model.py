import logging
import anndata as ad
import numpy as np
import scipy
import torch

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data  
import numpy as np
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_s 

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from sklearn import ensemble 

logging.basicConfig(level=logging.INFO)

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

from torch.autograd import Variable
import time

def idx2onehot(idx, n):

    idxlist = list(set(idx))
    d = {}
    for i in range(0,len(idxlist)):
        d[idxlist[i]] = i
    idx = torch.tensor([d[i] for i in idx])

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot

# TORCH MODEL
# activate function
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))

class Encoder(nn.Module):
    def __init__(self, in_dim=N_PCS, h1_dim=128, h2_dim=64, h3_dim=32, z_dim=Z_DIM):
        super(Encoder, self).__init__()
        self.relu_l = nn.ReLU(True)
        self.encoder = nn.Sequential(

            nn.Linear(in_dim, h1_dim),  
            nn.BatchNorm1d(h1_dim),
            Mish(),

            nn.Linear(h1_dim,h2_dim),  
            nn.BatchNorm1d(h2_dim),
            Mish(),
            
            nn.Linear(h2_dim, h3_dim),  
            nn.BatchNorm1d(h3_dim),
            Mish(),

            nn.Linear(h3_dim,z_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, z_dim=Z_DIM_New, h1_dim=32, h2_dim=64,h3_dim=128, out_dim=N_PCS):
        super(Decoder, self).__init__()
        self.relu_l = nn.ReLU(True)
        self.decoder = nn.Sequential(

            nn.Linear(z_dim, h1_dim),  
            nn.BatchNorm1d(h1_dim),
            Mish(),

            nn.Linear(h1_dim,h2_dim),  
            nn.BatchNorm1d(h2_dim),
            Mish(),
            
            nn.Linear(h2_dim, h3_dim),  
            nn.BatchNorm1d(h3_dim),
            Mish(),
            
            nn.Linear(h3_dim,out_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class cross_model(nn.Module):
    def __init__(self):
        super(cross_model, self).__init__()
        self.enc_gex = Encoder().cuda()
        self.dec_gex = Decoder().cuda()
        self.enc_atac = Encoder().cuda()
        self.dec_atac = Decoder().cuda()

    def forward(self, gex, atac, batch_information):
        self.z_gex = self.enc_gex(gex)
        self.z_atac = self.enc_atac(atac)

        z_gex_new = torch.cat((self.z_gex,batch_information), dim=-1)
        z_atac_new = torch.cat((self.z_atac,batch_information), dim=-1)

        atac_recon = self.dec_atac(z_gex_new)
        gex_recon = self.dec_gex(z_atac_new)
        return gex_recon, atac_recon

    def latent(self, gex, atac):
        z_gex = self.enc_gex(gex)
        z_atac = self.enc_atac(atac)
        return z_gex, z_atac

smth_loss = nn.SmoothL1Loss().cuda()
euc_loss = nn.MSELoss().cuda()

def print_epoch(epoch):
    return epoch<=5 or (epoch+1)%20==0

def print_epoch(epoch):
    return epoch<=5 or (epoch+1)%20==0

def evaluation_function(pred, true):
    pred = torch.tensor(pred).cuda()
    true = torch.tensor(true).cuda()
    final = np.array(torch.mul(pred,true).cpu().detach())
    return np.sum(final)/len(pred)

