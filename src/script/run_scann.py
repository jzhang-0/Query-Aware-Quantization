import sys 
sys.path.append("src/packages") 
from scann_lib import ScaNN_PQ,codebook_init
from evaluationRecall import recall_atN, SearchNeighbors_PQ

import argparse
import os
import numpy as np
np.random.seed(15)

sys.path.append("src") 

from args import *

from pprint import pprint

parser = argparse.ArgumentParser(description='Initialize Parameters!')

parser = parse_scann_van(parser)
config = parser.parse_args()

config = config_args_SpecData(config)

config.output_file = config.exp_path + "/save"

def main(config):
    config.suffix = f"PQ_M_{config.M}_K_{config.K}"

    dataset = np.load(config.data)
    n = dataset.shape[0]

    print(f"{config.suffix},bits:{config.M*np.log2(config.K)},T:{config.T},normalization:{config.nor},train number <= {config.train_num}")
    pprint(vars(config))

    if config.nor == 0:
        norm = np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        config.T = config.T*np.mean(norm) 
        
    s_pq = ScaNN_PQ(config.M, config.K, config.D, config.T, config.nor)

    C = codebook_init(dataset, config.M, config.K)
    max_iter = config.max_iter

    if n < config.train_num:
        C, S_ind, _ = s_pq.train(train_data = dataset, C = C, saveTR=0, maxiter=max_iter)
    else:
        train_idx = np.random.choice(n, config.train_num, replace = 0)
        C, _, _ = s_pq.train(train_data = dataset[train_idx], C = C, saveTR=0, maxiter=max_iter)
        weight_MA = s_pq.par_compute_H(dataset) 
        S_ind, _ = s_pq.index(C, dataset, weight_H=weight_MA, njobs=20)

    dirs = config.output_file
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.save(dirs+f"codebook_C_{config.suffix}",C)
    np.save(dirs+f"data_pqcodes_{config.suffix}",S_ind)

    print("The Anisotropic PQ is complete.")

    M = config.M
    Ks = config.K
    queries = np.load(config.queries)
    D = queries.shape[1]
    Ds = D // M

    code = S_ind
    gr_t100 = np.load(config.tr100)

    sn_pq = SearchNeighbors_PQ(M=M, Ks=config.K, D=D, pq_codebook = C.reshape(M, Ks, Ds), pq_codes = code, metric="dot_product")

    neighbors = sn_pq.par_neighbors(queries, config.topk, njobs=20)

    recall_atN(neighbors, gr_t100)

if __name__ == "__main__":
    main(config)
