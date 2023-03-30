from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from run_pq import parse_args

import numpy as np
np.random.seed(123)


if __name__ == '__main__':
    codebook = 8
    Ks = 16
    metric = 'product'

    X = np.load("data/EchoNest/EchoNest_data.npy")
    Q = np.load("data/EchoNest/queries.npy")
    G = np.load("data/EchoNest/true_neighbors_top100.npy")
    G = G[:,0].reshape(-1,1)
        
    a_ = np.random.choice(range(X.shape[0]),X.shape[0],replace=0)
    T = X[a_[0:100000]]

    quantizer = NormPQ(n_percentile=Ks, quantize=PQ(M=codebook-1, Ks=Ks))
    execute(quantizer,  X, T, Q, G, metric)
