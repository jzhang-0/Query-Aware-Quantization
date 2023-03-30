import numpy as np
# from scipy import special as spc
import scipy.spatial.distance as dist

def ALSHDataTransform(data, queries, m):
    data_trans = np.array(list(map(lambda x: np.append(x, \
        [np.linalg.norm(x)**(2*(k+1)) for k in range(m)]), data)))
    queries_trans = np.array(list(map(lambda x: np.append(x, \
        [.5 for k in range(m)]), queries)))
    return (data_trans, queries_trans)

#random "a" values of dimensions of the data
def hashedFunctions(N_Hashs, dim):
    return np.random.normal(size=(N_Hashs, dim))

#returns (a^T.x + b)/r of datapoints
# val is b ~U[0,r]
# rad = r
def dataToHashFunctions(datapts, hash_raw, rad):
    hashed = np.dot(datapts, hash_raw.T)
    val = rad * np.random.random(1)
    return np.floor((hashed + val) / rad)


#returns pairwise hamming distances and their sorted counter parts
# def hammDist(hash_pt1, hash_pt2):
#     hDist = dist.cdist(hash_pt1, hash_pt2, metric='hamming')
#     hDist = 1.0 - hDist
#     return (hDist, np.argsort(hDist, axis=1))

def _sort_topk_adc_score(adc_score, topk):
    ind = np.argpartition(adc_score, topk)[0:topk]
    return ind[np.argsort(adc_score[ind])]

def neighbors(h1, h2):
    """
    h1:(n_d,K)
    h2:(n_q,K)
    """
    hDist = dist.cdist(h1, h2, metric='hamming') # shape=(n_d,n_q)
    return (np.argsort(hDist, axis=0)).T


def normalization(data):
    """
    data:shape=(n,d)
    """
    return data / np.linalg.norm(data,axis=1)[:,np.newaxis]

def scaling_data(data):
    """
    data:shape=(n,d)
    """
    data_norms = np.linalg.norm(data, axis=1)
    data_max_norm = np.max(data_norms)
    data /= data_max_norm
    return data 

def simple_lsh_transform(data,queries):
    """
    data:shape=(n,d)
    queries = (nq,d)
    """
    data_norms = np.linalg.norm(data, axis=1)
    r_norms = (1 - data_norms**2)**(1/2)
    data_trans = np.c_[data,r_norms]

    m = np.zeros(queries.shape[0])
    queries_trans = np.c_[queries,m]
    return data_trans,queries_trans

def compute_recall(neighbors, ground_truth):
    total = 0
    for gt_row, row in zip(ground_truth, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / ground_truth.size 
