

import argparse
import numpy as np

from utils import simple_lsh_transform,scaling_data,normalization,neighbors,compute_recall
from simplelsh import SimpleLsh

parser = argparse.ArgumentParser(description='Initialize Parameters!')

parser.add_argument('--data', default='data/LastFM/LastFM_data32D.npy', type=str, help='path of datafile')
parser.add_argument('--queries', default='data/LastFM/queries_32D.npy', type=str, help='path of datafile')
parser.add_argument('--true_neighbors_top100', default='data/LastFM/true_neighbors_top100_32D.npy', type=str, help='path of datafile')

parser.add_argument('--K', default=32, type=int, help='the number of hash function')

config = parser.parse_args()

def simple_lsh(config):
    print(f"hash function number : K={config.K}")

    dataset = np.load(config.data)
    queries = np.load(config.queries)
    tr100 = np.load(config.true_neighbors_top100)
    tr1 = tr100[:,0]
    tr10 = tr100[:,0:10]

    dataset = scaling_data(dataset)
    queries = normalization(queries)
    dataset,queries = simple_lsh_transform(dataset,queries)

    dim = dataset.shape[1]
    sl = SimpleLsh(config.K,dim)

    h1 = sl.hash_codes(dataset)
    h2 = sl.hash_codes(queries)

    neighbors_m = neighbors(h1, h2)

    rl = [1,2,4,8,16,20,32,64,128,256,512,1024]

    for i in rl:
        print(f"recall 1@{i} = {compute_recall(neighbors_m[:,0:i],tr1)}")

    rl = [10,16,20,32,64,128,256,512,1024]
    for i in rl:
        print(f"recall 10@{i} = {compute_recall(neighbors_m[:,0:i],tr10)}")


if __name__ == "__main__":
    simple_lsh(config)