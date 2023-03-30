import sys
sys.path.append("src/packages")

from QUIP import QUIP_cov, get_CovMA


from pprint import pprint

import argparse
import os
import numpy as np

from args import *

from evaluationRecall import SearchNeighbors_PQ, recall_atN


parser = argparse.ArgumentParser(description='Initialize Parameters!')
parser = parse_QUIP_cov_van(parser)

args = parser.parse_args()
args = config_args_SpecData(args)
args = ex_config(args)
save_path = args.save_path

pprint(args)

dataset = np.load(args.data) 
queries = np.load(args.queries) 
gr_t100 = np.load(args.tr100)  

sample_queries = np.load(args.sample_queries) 
sample_num = args.sample_num 
sample_queries = sample_queries[0:sample_num,:]

M, Ks=args.M, args.Ks
D = dataset.shape[1]
Ds = D // M


X = np.array(dataset, dtype=np.float32)
CovMA = get_CovMA(sample_queries, M)
qc = QUIP_cov(M,Ks,D)

if X.shape[0] > args.training_sample_size:
    a_ = np.random.choice(range(X.shape[0]),X.shape[0],replace=0)
    train_X = X[a_[0:args.training_sample_size]]
    
    qc.train(train_X, CovMA)
    X_code = qc.encode(X, CovMA)

else:
    train_X = X
    X_code = qc.train(train_X, CovMA)


sn_pq = SearchNeighbors_PQ(M = M, Ks = Ks, D=D, pq_codebook = qc.pq_codebook, pq_codes = X_code, metric="dot_product")

neighbors = sn_pq.par_neighbors(queries, 512, os.cpu_count()//2)

recall_atN(neighbors, gr_t100)


suffix = f"M{M}_K{Ks}_datan_{args.datan}_samplesize{args.training_sample_size}_QUIP"
np.save(f"{save_path}/pq_code_{suffix}", X_code)
np.save(f"{save_path}/pq_codebook_{suffix}", qc.pq_codebook)