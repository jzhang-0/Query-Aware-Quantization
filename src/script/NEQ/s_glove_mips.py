from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from run_pq import parse_args

import argparse

args = argparse.ArgumentParser()
args.add_argument("--M", default= 25, type=int)
config = args.parse_args()

import numpy as np
np.random.seed(123)


if __name__ == '__main__': 
    codebook = config.M
    Ks = 16
    metric = 'product'

    X = np.load("data/glove_mips/glove.npy")
    Q = np.load("data/glove_mips/queries_test.npy")
    G = np.load("data/glove_mips/tr100_test.npy")
    G = G[:,0].reshape(-1,1)
        
    a_ = np.random.choice(range(X.shape[0]),X.shape[0],replace=0)
    T = X[a_[0:100000]]

    quantizer = NormPQ(n_percentile=Ks, quantize=PQ(M=codebook-1, Ks=Ks))
    execute(quantizer,  X, T, Q, G, metric)
