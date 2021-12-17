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

# for the datasets we use, please refer: https://openproblems.bio/neurips_docs/submission/quickstart/
par1 = {
    'input_train_mod1': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_sol.h5ad',
    'input_test_mod1': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad',
    'input_test_sol':'./output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_sol.h5ad',
    'distance_method': 'minkowski',
    'output': './output/predictions/match_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.method.output.h5ad',
    'n_pcs': 4,
    'n_neighbors': 1000,
}

par2 = {
    'input_train_mod1': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_sol.h5ad',
    'input_test_mod1': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod2.h5ad',
    'input_test_sol': './output/datasets/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_sol.h5ad',
    'distance_method': 'minkowski',
    'output': './output/predictions/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.method.output.h5ad',
    'n_pcs': 4,
    'n_neighbors': 1000,
}

par3 = {
    'input_train_mod1': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_sol.h5ad',
    'input_test_mod1': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad',
    'input_test_sol': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_sol.h5ad',
    'distance_method': 'minkowski',
    'output': './output/predictions/match_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.method.output.h5ad',
    'n_pcs': 4,
    'n_neighbors': 1000,
}

par4 = {
    'input_train_mod1': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_sol.h5ad',
    'input_test_mod1': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod2.h5ad',
    'input_test_sol': './output/datasets/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_sol.h5ad',
    'distance_method': 'minkowski',
    'output': './output/predictions/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.method.output.h5ad',
    'n_pcs': 4,
    'n_neighbors': 1000,
}

parlist = [par1,par2,par3,par4]

def evaluation_function(pred, true):
    pred = torch.tensor(pred).cuda()
    true = torch.tensor(true).cuda()
    final = np.array(torch.mul(pred,true).cpu().detach())
    return np.sum(final)/len(pred)

# TODO: change this to the name of your method
for i in range(0,4):
    par = parlist[i]
    method_id = "autoencoder"

    logging.info('Reading `h5ad` files...')
    input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
    input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
    input_train_sol = ad.read_h5ad(par['input_train_sol'])
    input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
    input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

    order = np.argsort(input_train_sol.uns['pairing_ix'])
    input_train_mod2 = input_train_mod2[order, :]

    logging.info('Reading `h5ad` files...')
    input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
    input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
    input_train_sol = ad.read_h5ad(par['input_train_sol'])
    input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
    input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

    order = np.argsort(input_train_sol.uns['pairing_ix'])
    input_train_mod2 = input_train_mod2[order, :]

    # TODO: implement own method

    # This starter kit is split up into several steps.
    # * compute dimensionality reduction on [train_mod1, test_mod1] data
    # * train linear model to predict the train_mod2 data from the dr_mod1 values
    # * predict test_mod2 matrix from model and test_mod1
    # * calculate k nearest neighbors between test_mod2 and predicted test_mod2
    # * transform k nearest neighbors into a pairing matrix

    input_mod1 = ad.concat(
        {"train": input_train_mod1, "test": input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-"
    )

    input_mod2 = ad.concat(
        {"train": input_train_mod2, "test": input_test_mod2},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-"
    )

    # TODO: implement own method
    # Do PCA on the input data
    N_PCS = min(par['n_pcs'], int(0.8*input_mod1.shape[1]), int(0.8*input_mod2.shape[1]))
    N_PCS_new = N_PCS+10
    logging.info('Performing dimensionality reduction on modality 1 values...')
    embedder_mod1 = TruncatedSVD(n_components=N_PCS)
    mod1_pca = embedder_mod1.fit_transform(input_mod1.X)
    logging.info('Performing dimensionality reduction on modality 2 values...')
    embedder_mod2 = TruncatedSVD(n_components=N_PCS)
    mod2_pca = embedder_mod2.fit_transform(input_mod2.X)


    # split dimred back up
    X_train = mod1_pca[input_mod1.obs['group'] == 'train']
    X_test = mod1_pca[input_mod1.obs['group'] == 'test']
    Y_train = mod2_pca[input_mod2.obs['group'] == 'train']
    Y_test = mod2_pca[input_mod2.obs['group'] == 'test']

    assert len(X_train) + len(X_test) == len(mod1_pca)
    assert len(Y_train) + len(Y_test) == len(mod1_pca)

#     Z_DIM = par['z_dim']
    Z_DIM = 4
    Z_DIM_New = Z_DIM+10

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

    # parameters
#     EPOCH = par['epochs']
#     MAX_ITER = X_train.shape[0]
#     batch = par['batch']
#     lr = par['lr']
#     lambda_atac = par['lambda_atac']
#     lambda_latent = par['lambda_latent']


    #test
    EPOCH =250
    MAX_ITER = X_train.shape[0]
    batch =128
    lr = 0.001
    lambda_atac = 2
    lambda_latent = 100

    X_train, Y_train, X_test, Y_test = [
        Variable(torch.FloatTensor(x)).cuda() for x in [X_train, Y_train, X_test, Y_test]
    ]

    batch_info = idx2onehot(input_train_mod2.obs['batch'], 10).cuda()

    model = cross_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    ####################training steps #######################
    for epoch in range(EPOCH):
        if print_epoch(epoch): 
            logging.info('epoch: {}'.format(epoch+1))
            batch_information = batch_info
            X_recon, Y_recon = model(X_train, Y_train,batch_information)
            error_X = smth_loss(X_recon, X_train)
            error_Y = lambda_atac * smth_loss(Y_recon, Y_train)
            error_Z = lambda_latent * euc_loss(model.z_gex, model.z_atac)
            error = error_X + error_Y + error_Z
            logging.info('{}. {}. {}. {}\n'.format(error_X.item(), error_Y.item(), error_Z.item(), error.item()))

        train_idx = np.arange(len(X_train)); np.random.shuffle(train_idx)
        for time in range(0,MAX_ITER,batch):
            X_train_batch = X_train[train_idx[time:time+batch]]
            Y_train_batch = Y_train[train_idx[time:time+batch]]
            batch_information = batch_info[train_idx[time:time+batch]]
            optimizer.zero_grad()
            X_recon, Y_recon = model(X_train_batch, Y_train_batch, batch_information)
            z_X = model.z_gex
            z_Y = model.z_atac

            error_X = euc_loss(X_recon, X_train_batch)
            error_Y = lambda_atac * euc_loss(Y_recon, Y_train_batch)
            error_Z = lambda_latent * euc_loss(model.z_gex, model.z_atac)
#             error_X = -torch.cosine_similarity(X_recon, X_train_batch, dim=1)
#             error_Y = -torch.cosine_similarity(Y_recon, Y_train_batch, dim=1)
#             error_Z = -torch.cosine_similarity(model.z_gex, model.z_atac, dim=1)

            error = torch.mean(error_X) + torch.mean(error_Y) + torch.mean(error_Z)

            error.backward()
            optimizer.step()



    # testing

    def get_sigma(dists, k=10):
        sigma = np.sort(dists, axis=1)
        sigma = sigma[:,:k].flatten()
        sigma = np.sqrt(np.mean(np.square(sigma)))
        return sigma

    def dist2prob(row, k, sigma):
        idx = np.argsort(row)
        row[idx[k:]] = 0
        probs = row[idx[:k]]
        probs = -np.square(probs)/(2*sigma**2)
        probs = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(probs - logsumexp(probs))
        probs /= sum(probs)
        row[idx[:k]] = probs
        return row

    def kNN_probs(dists, k): 
        XY_dists = dists.copy()
        X2Y = np.apply_along_axis(dist2prob, 1, XY_dists, k, get_sigma(XY_dists,k))
        return X2Y

    model.eval()

    Z_gex_test = model.enc_gex(X_test).cpu().detach().numpy()
    Z_atac_test = model.enc_atac(Y_test).cpu().detach().numpy()

    k = par['n_neighbors']
    dists = euclidean_distances(Z_gex_test, Z_atac_test)
    pairing_matrix = kNN_probs(dists, k)
    pairing_matrix = csr_matrix(pairing_matrix)

    logging.info('write prediction output')
    out = ad.AnnData(
        X=pairing_matrix,
        uns={
            "dataset_id": input_train_mod1.uns["dataset_id"],
            "method_id": method_id
        }
    )
    out.write_h5ad(par['output'], compression="gzip")
    ##########################################################
    input_test_sol = ad.read_h5ad(par['input_test_sol'])
    print(evaluation_function(out.X.todense(), input_test_sol.X.todense()))

