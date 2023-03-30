import sys
sys.path.append("src/packages")
from evaluationRecall import recall_atN, SearchNeighbors_PQ
from PQ import PQ

import argparse
import os

import numpy as np
from pprint import pprint
np.random.seed(15)

from args import *

def main():
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser = parse_PQ_van(parser)

    args = parser.parse_args()
    args = config_args_SpecData(args)
    args = ex_config(args)
    save_path = args.save_path

    n_train = args.training_sample_size


    pprint(args)


    dataset = np.load(args.data) 
    queries = np.load(args.queries) 
    gr_t100 = np.load(args.tr100)


    X = dataset

    M, Ks=args.M, args.Ks
    D = dataset.shape[1]
    pq = PQ(M=M,Ks=Ks)

    X= np.array(X,dtype=np.float32)

    if X.shape[0] <= n_train:
        pq.fit(X)
    else:
        pq.fit(X[0:n_train])

    X_code = pq.encode(X)  


    sn_pq = SearchNeighbors_PQ(M=M, Ks=Ks, D=D, pq_codebook = pq.codewords, pq_codes = X_code, metric="dot_product")

    neighbors = sn_pq.par_neighbors(queries, args.topk, os.cpu_count()//2)

    recall_atN(neighbors,gr_t100)

    suffix = f"M{M}_K{Ks}_datan_{args.datan}_samplesize{n_train}_PQ"
    np.save(f"{save_path}/pq_code_{suffix}", X_code)
    np.save(f"{save_path}/pq_codebook_{suffix}", pq.codewords)

if __name__ == "__main__":
    main()