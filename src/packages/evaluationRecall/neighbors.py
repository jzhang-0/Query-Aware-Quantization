import os
import numpy as np
from joblib import Parallel, delayed, dump,load
from .utils import timefn,for_all_methods
njobs = os.cpu_count()//2

def compute_recall(neighbors, ground_truth):
    total = 0
    for gt_row, row in zip(ground_truth, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return round(total / ground_truth.size, 4)

def recall_atN_(neighbors_matrix, ground_truth):
    """
    Args:
        neighbors_matrix:(nq,topK=512)
        ground_truth: 1d or 2d
    """
    topk = neighbors_matrix.shape[1]
    if ground_truth.ndim == 1:
        ng = 1
    if ground_truth.ndim == 2:
        ng = ground_truth.shape[1]

    N = [1,2,4,8,10,16,20,32,64,100,128,256,512]

    recall_list=[]
    N_list = []
    for tn in N:
        if ng <= tn <= topk:
            neighbors_matrix_topN = neighbors_matrix[:,0:tn]
            recall = compute_recall(neighbors_matrix_topN, ground_truth)
            recall_list.append(recall)
            N_list.append(tn)
            print(f"recall {ng}@{tn} = {recall}")
    return recall_list,N_list

def recall_atN(neighbors_matrix, ground_truth):
    """
    Args:
        neighbors_matrix:(nq, topK=512)
        ground_truth:(nq,) or (nq,>=10)
    """
    if ground_truth.ndim == 2 :
        ground_truth_1 = ground_truth[:,0]
        ground_truth_10 = ground_truth[:,0:10]
        r1, N1 = recall_atN_(neighbors_matrix,ground_truth_1)
        print("\n")
        r10, N10 = recall_atN_(neighbors_matrix,ground_truth_10)
        print("\n")
        print(f"N={N1}")
        print(f"recall1@N:{r1}")
        print(f"N={N10}")
        print(f"recall10@N:{r10}")

    if ground_truth.ndim == 1 :
        r1, N1 = recall_atN_(neighbors_matrix, ground_truth)
        print("\n")
        print(f"N={N1}")
        print(f"recall1@N:{r1}")


class SearchNeighbors:
    def __init__(self, metric) -> None:
        self.metric = metric
        if metric != "dot_product" and metric != "l2_distance":
            raise Exception(f"not suport {metric},optional:l2_distance or dot_product")

    def compute_recall(self, neighbors, ground_truth):
        # total = 0
        # for gt_row, row in zip(ground_truth, neighbors):
        #     total += np.intersect1d(gt_row, row).shape[0]
        # return total / ground_truth.size
        return compute_recall(neighbors,ground_truth)

    def brute_force_search(self, target_set, test_set, metric="l2_distance"):
        if metric != "dot_product" and metric != "l2_distance":
            raise Exception(f"not suport {metric},optional:l2_distance or dot_product")

        if metric == "dot_product":
            inner_product = target_set @ test_set.T
            return np.argmax(inner_product, axis=0)

        if metric == "l2_distance":
            q_2 = np.linalg.norm(test_set, axis=1).reshape(-1, 1) ** 2
            x_2 = np.linalg.norm(target_set, axis=1).reshape(1, -1) ** 2

            q_x = test_set @ target_set.T
            gt_distance = q_2 - 2 * q_x + x_2

            gt = np.argmin(gt_distance, axis=1)
            return gt

    # @jit
    def _sort_topk_adc_score(self, adc_score, topk):
        metric = self.metric

        if metric == "dot_product":
            ind_t = np.argpartition(adc_score, -topk)
            ind = ind_t[-topk:]
            t1 = np.argsort(adc_score[ind])
            t2 = ind[t1]
            t3 = np.flip(t2)
            return t3

        if metric == "l2_distance":
            ind = np.argpartition(adc_score, topk)[0:topk]
            return ind[np.argsort(adc_score[ind])]

# @for_all_methods(profile)
class SearchNeighbors_PQ(SearchNeighbors):
    """
    For the PQ indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.
    All vectors must be np.ndarray 
    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
    
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        D (int): The dim of each vector
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        metric (str): dot_product or l2_distance        
    
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        metric (str): dot_product or l2_distance
    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, metric="l2_distance") -> None:
        SearchNeighbors.__init__(self, metric)
        self.M = M
        self.Ks = Ks
        self.D = D
        assert D % M == 0, "D must be divisible by M"
        self.Ds = D // M

        self.pq_codebook = pq_codebook
        self.pq_codes = pq_codes
        assert pq_codebook.shape == (M, Ks, self.Ds), "pq_codebook.shape must equal to (M,Ks,Ds)"

        self.tem_value = np.zeros(shape=(pq_codes.shape))
        self.index = np.zeros_like(pq_codes)

        for i in range(M):
            self.index[:,i] = i

    # @jit
    def compute_distance(self, query):
        """
        The distances (the squared Euclidean distance or inner product) are computed by comparing each sub-vector of
        the query to the codewords for each sub-subspace. `lookup_table[m][ks]` contains the squared Euclidean
        distance or inner product between the `m`-th sub-vector of the query and the `ks`-th codeword for the `m`-th
        sub-space (`self.codewords[m][ks]`). By looking up table to compute distance.
        Args:
            query (np.ndarray): Input vector with shape=(D, ) 
        Returns:
            np.ndarray: Asymmetric Distances with shape=(n, )  
        """
        pq_codebook = self.pq_codebook
        codes = self.pq_codes
        metric = self.metric

        M = self.M
        Ds = self.Ds

        q = query.reshape(M, Ds)
        if metric == "dot_product":
            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            # table_vaule = lookup_table[range(M), codes] # (n,M)
            # inner_prod = np.sum(table_vaule, axis=1)

            self.tem_value  = lookup_table[self.index, codes]
            inner_prod = np.sum(self.tem_value, axis=1)
            return inner_prod

        if metric == "l2_distance":
            lookup_table = np.linalg.norm(pq_codebook - q[:, np.newaxis, :], axis=2) ** 2
            dists = np.sum(lookup_table[range(M), codes], axis=1)
            return dists

    def search_neighbors(self, query, topk):
        """
        Args:
            pq_codebook, codes, query, metric are the same as method compute_distance
            topk (int): this method will return topk neighbors
        Returns:
            index (np.darray): query's topk neighbors  
        """
        adc_score = self.compute_distance(query)
        return self._sort_topk_adc_score(adc_score, topk)

    @timefn
    def neighbors(self, queries, topk):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)  
        
        """
        assert queries.ndim == 2
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neighbors(q, topk)

        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix

    @timefn
    def par_neighbors(self, queries, topk, njobs=njobs):
        """
        Parallel version of method neighbors
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query
            njobs: the process numbers use
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)  
        
        """
        assert queries.ndim == 2
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)

        result = Parallel(n_jobs = njobs,backend='multiprocessing')(delayed(self.search_neighbors)(q,topk) for q in queries)

        for i in range(n):
            neighbors_matrix[i] = result[i]


        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix


    def pq_recall(self, queries, topk, ground_truth):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query.
            ground_truth(np.darray): shape=(nq,) or (nq, num), queries real neighbors (top1 or top num)
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)
        """
        ground_truth = np.array(ground_truth)

        try:
            if topk > self.neighbors_matrix.shape[1]:
                neighbors_matrix = self.neighbors(queries, topk)
            else:
                neighbors_matrix = self.neighbors_matrix[:, 0:topk]
        except AttributeError:
            neighbors_matrix = self.neighbors(queries, topk)

        recall = self.compute_recall(neighbors_matrix, ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2:
            ng = ground_truth.shape[1]

        print(f"recall {ng}@{nr} = {recall}")


class SearchNeighbors_AQ(SearchNeighbors):
    """
    Babenko, Artem, and Victor Lempitsky. "Additive quantization for extreme vector compression." CVPR 2014.
    
    Args:
        M (int): The number of codebooks
        K (int): The number of codewords for each codebook
        D (int): The dim of each vector
        aq_codebooks (np.ndarray): shape=(M*K, D) with dtype=np.float32.
            aq_codebooks[0:K,:] represents the K codewords in the first codebook
            aq_codebooks[(m-1)*K:mK,:] represents the K codewords in the m-th codebook
        aq_codes (np.ndarray): AQ codes with shape=(n, M) and dtype=np.int, where n is the number of encoded datapoints
            aq_codes[i,j] is in {0,1,...,K-1} for all i,j 
        metric (str): dot_product or l2_distance        
    
    Attributes:
        M (int): The number of codebooks
        K (int): The number of codewords for each codebook
        D (int): The dim of each vector
        aq_codebooks (np.ndarray): shape=(M*K, D) with dtype=np.float32.
        aq_codes (np.ndarray): AQ codes with shape=(n, M) and dtype=np.int, where n is the number of encoded datapoints
        metric (str): dot_product or l2_distance
        norm_sq (np.ndarray):shape=(n,), the square of the length of the quantized datapoints is recorded in norm_sq. 
    """

    def __init__(self, M, K, D, aq_codebooks, aq_codes, metric="l2_distance") -> None:
        SearchNeighbors.__init__(self, metric)
        assert aq_codebooks.shape == (M * K, D), "aq_codebooks.shape must equal to (M*K, D)"
        assert aq_codes.shape[1] == M, "aq_codebooks.shape must equal to (n, M)"

        self.M = M
        self.K = K
        self.D = D
        self.aq_codebooks = aq_codebooks
        self.aq_codes = aq_codes

        self.aq_codes_2 = aq_codes + np.arange(M) * K

        if self.metric == "l2_distance":
            n = aq_codes.shape[0]
            self.norm_sq = np.zeros(n)
            for i in range(n):
                x = np.sum(aq_codebooks[self.aq_codes_2[i]], axis=0)
                self.norm_sq[i] = np.vdot(x, x)

    def compute_distance(self, query):
        """
        Compute the distances (the squared Euclidean distance or inner product).
        Args:
            query (np.ndarray): Input vector with shape=(D, ) 
        Returns:
            np.ndarray: Asymmetric Distances with shape=(n, )  
        """
        aq_codebooks = self.aq_codebooks
        codes_2 = self.aq_codes_2
        metric = self.metric

        if metric == "dot_product":
            lookup_table = aq_codebooks @ query
            inner_prod = np.sum(lookup_table[codes_2], axis=1)

            return inner_prod

        if metric == "l2_distance":
            lookup_table = aq_codebooks @ query
            inner_prod = np.sum(lookup_table[codes_2], axis=1)

            dists = np.vdot(query, query) + self.norm_sq - 2 * inner_prod
            return dists

    def search_neighbors(self, query, topk):
        """
        Args:
            topk (int): this method will return topk neighbors
        Returns:
            index (np.darray): query's topk neighbors for a query.
        """
        adc_score = self.compute_distance(query)
        return self._sort_topk_adc_score(adc_score, topk)

    def neighbors(self, queries, topk):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query.
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)          
        """
        assert queries.ndim == 2
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neighbors(q, topk)

        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix

    @timefn
    def par_neighbors(self, queries, topk, njobs=njobs):
        """
        Parallel version of method neighbors
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query
            njobs: the process numbers use
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)  
        
        """
        assert queries.ndim == 2
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)

        result = Parallel(n_jobs = njobs,backend='multiprocessing')(delayed(self.search_neighbors)(q,topk) for q in queries)

        for i in range(n):
            neighbors_matrix[i] = result[i]


        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix

    def aq_recall(self, queries, topk, ground_truth):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
            topk (int): this method will return topk neighbors for each query.
            ground_truth(np.darray): shape=(nq,) or (nq, num), queries real neighbors (top1 or top num)
        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)
        """
        ground_truth = np.array(ground_truth)

        try:
            if topk > self.neighbors_matrix.shape[1]:
                neighbors_matrix = self.neighbors(queries, topk)
            else:
                neighbors_matrix = self.neighbors_matrix[:, 0:topk]
        except AttributeError:
            neighbors_matrix = self.neighbors(queries, topk)

        recall = self.compute_recall(neighbors_matrix, ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2:
            ng = ground_truth.shape[1]

        print(f"recall {ng}@{nr} = {recall}")