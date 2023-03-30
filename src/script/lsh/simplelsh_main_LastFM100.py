

import argparse
import numpy as np

from utils import simple_lsh_transform,scaling_data,normalization,neighbors,compute_recall
from simplelsh import SimpleLsh

parser = argparse.ArgumentParser(description='Initialize Parameters!')

parser.add_argument('--data', default='data/lastfmdata/lastfmdata_items_100D.npy', type=str, help='path of datafile')
parser.add_argument('--queries', default='data/lastfmdata/queries_100D.npy', type=str, help='path of datafile')
parser.add_argument('--true_neighbors_top100', default='data/lastfmdata/true_neighbors_top100.npy', type=str, help='path of datafile')

parser.add_argument('--K', default=100, type=int, help='the number of hash function')

config = parser.parse_args()

def simple_lsh(config):
    print("max inner search")
    print(f"hash function number : K={config.K}")

    dataset = np.load(config.data)
    queries = np.load(config.queries)
    # tr100 = np.load(config.true_neighbors_top100)

    inner = queries@dataset.T
    tr = np.fliplr(np.argsort(inner,axis=1))
    tr100 = tr[:,0:100]

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
    print(rl)
    r1_list = []
    for i in rl:
        recall1 = compute_recall(neighbors_m[:,0:i],tr1)
        r1_list.append(recall1)
        print(f"recall 1@{i} = {recall1}")

    rl = [10,16,20,32,64,128,256,512,1024]
    for i in rl:
        print(f"recall 10@{i} = {compute_recall(neighbors_m[:,0:i],tr10)}")

    print(f"recall1@N : {r1_list}")

if __name__ == "__main__":
    simple_lsh(config)