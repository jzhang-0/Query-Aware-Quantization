from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from run_pq import parse_args

import numpy as np
np.random.seed(123)


if __name__ == '__main__':
    codebook = 25
    Ks = 16
    metric = 'product'

    X = np.load("data/LastFM100D/LastFM_data100D.npy")
    Q = np.load("data/LastFM100D/queries_100D.npy")
    G = np.load("data/LastFM100D/true_neighbors_top100_100D.npy")
    G = G[:,0].reshape(-1,1)
        
    a_ = np.random.choice(range(X.shape[0]),X.shape[0],replace=0)
    T = X[a_[0:100000]]

    quantizer = NormPQ(n_percentile=Ks, quantize=PQ(M=codebook-1, Ks=Ks))
    execute(quantizer,  X, T, Q, G, metric)
