import numpy as np
from .neighbors import SearchNeighbors_AQ, SearchNeighbors_PQ
from .utils import timefn, for_all_methods

class SearchNeighbors_PQIVF(SearchNeighbors_PQ):
    """
    the asymmetric distance computation combined with an inverted file system (IVFADC) 
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
        vq_code_book (np.ndarray): shape=(k', D) with dtype=np.float32.
            vq_code_book[m] means m-th codeword (D-dim)
        vq_codes (np.ndarray): VQ codes with shape=(n, ) and dtype=np.int
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        metric (str): dot_product or l2_distance        
    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, vq_code_book, vq_codes, metric) -> None:
        super().__init__(M, Ks, D, pq_codebook, pq_codes, metric=metric)
        self.vq_codes = vq_codes
        self.vq_code_book = vq_code_book

        k_v = vq_code_book.shape[0]
        self.vq_cluster = [[] for i in range(k_v)]
        data_index = 0
        for i in vq_codes:
            self.vq_cluster[i].append(data_index)
            data_index += 1

    def dataIndex_tosearch(self, cluster_id):
        index = []
        for i in cluster_id:
            index += self.vq_cluster[i]
        return index

    # @profile
    def _vq(self, query, num_centroids_to_search):
        if self.metric == "dot_product":
            inner_M = self.vq_code_book @ query
            c_id = np.argsort(inner_M)[-num_centroids_to_search:]
            c_id = np.flip(c_id)

            inner_1 = inner_M[c_id]

            return c_id, inner_1

        if self.metric == "l2_distance":
            dist = np.linalg.norm(query - self.vq_code_book, axis=1) ** 2
            c_id = np.argsort(dist)[0:num_centroids_to_search]

            dist1 = dist[c_id]
            return c_id, dist1

    # @profile
    def search_neighbors_IVFADC(self, query, num_centroids_to_search, topk=64):
        metric = self.metric
        pq_codebook = self.pq_codebook
        pq_codes = self.pq_codes

        M = self.M
        Ds = self.Ds

        cluster_id, adc1 = self._vq(query, num_centroids_to_search)
        index = self.dataIndex_tosearch(cluster_id)

        adc = np.zeros(len(index))
        if metric == "dot_product":
            q = query.reshape(M, Ds)
            adc = np.array(
                [adc1[i] for i in range(num_centroids_to_search) for j in range(len(self.vq_cluster[cluster_id[i]]))])

            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            inner_prod_2 = np.sum(lookup_table[range(M), pq_codes[index, :]], axis=1)
            adc = adc + inner_prod_2

        if metric == "l2_distance":
            i1 = 0
            i2 = 0
            for i in cluster_id:
                q_i = query - self.vq_code_book[i]
                q_i = q_i.reshape(M, Ds)

                lookup_table = np.linalg.norm(pq_codebook - q_i[:, np.newaxis, :], axis=2) ** 2
                dists = np.sum(lookup_table[range(M), pq_codes[self.vq_cluster[i], :]], axis=1)

                i2 += len(self.vq_cluster[i])
                adc[i1:i2] = dists
                i1 = i2
        index = np.array(index)

        if index.shape[0] < topk:
            index_ = np.full(topk, fill_value=-1, dtype=int)
            if metric == "dot_product":
                ind = np.argsort(adc)
                index_[0:index.shape[0]] = np.flip(index[ind])
                return index_
            if metric == "l2_distance":
                ind = np.argsort(adc)
                index_[0:index.shape[0]] = index[ind]
                return index_
        else:
            return index[self._sort_topk_adc_score(adc, topk)]

    @timefn
    def neighbors_ivf(self, queries, num_centroids_to_search, topk):
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neighbors_IVFADC(q, num_centroids_to_search, topk)

        return neighbors_matrix


    def ex_init(self, queries):
        self.k_v = self.vq_code_book.shape[0]
        inv_tab = [[] for i in range(self.k_v)]

        data_index = 0
        for i in self.vq_codes:
            inv_tab[i].append(data_index)  
            data_index += 1

        self.inv_tab = inv_tab


        self.queries = queries

        l = len(inv_tab)
        self.count = np.zeros(l)
        for i in range(l):
            self.count[i] = len(inv_tab[i])

        inner_M = queries @ self.vq_code_book.T 
        c_id = np.argsort(inner_M, axis=1)
        self.c_id = np.fliplr(c_id)

        row = np.arange(queries.shape[0])[:, np.newaxis]
        self.inner_1 = inner_M[row, self.c_id]

    #TODO: improved
    def auto_centroid(self, i, fix_num):
        q = self.queries[i]
        qc_id = self.c_id[i]
        num=0
        
        search_cid = []
        for id in qc_id:
            search_cid.append(id)
            num += self.count[id]
            if num > fix_num:
                break
        return len(search_cid)

    @timefn
    def neighbors_ivf_fixnum(self, queries, num_to_search, topk):
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]

            num_centroids_to_search = self.auto_centroid(i, num_to_search)
            neighbors_matrix[i] = self.search_neighbors_IVFADC(q, num_centroids_to_search, topk)

        return neighbors_matrix




    def pqivf_recall(self, queries, num_centroids_to_search, topk, ground_truth):
        ground_truth = np.array(ground_truth)

        neighbors_matrix = self.neighbors_ivf(queries, num_centroids_to_search, topk)
        recall = self.compute_recall(neighbors_matrix, ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2:
            ng = ground_truth.shape[1]

        print(f"recall {ng}@{nr} = {recall}")

class SearchNeighbors_AQIVF(SearchNeighbors_AQ):
    def __init__(self, M, K, D, aq_codebooks, aq_codes, vq_codebook, vq_codes, queries, metric="l2_distance") -> None:
        super().__init__(M, K, D, aq_codebooks, aq_codes, metric=metric)

        self.k_v = vq_codebook.shape[1]
        inv_tab = [[] for i in range(self.k_v)]

        data_index = 0
        for i in vq_codes:
            inv_tab[i].append(data_index)  
            data_index += 1

        self.inv_tab = inv_tab
        self.code_book = vq_codebook
        self.queries = queries

        l = len(inv_tab)
        self.count = np.zeros(l)
        for i in range(l):
            self.count[i] = len(inv_tab[i])

        inner_M = queries @ vq_codebook
        c_id = np.argsort(inner_M, axis=1)
        self.c_id = np.fliplr(c_id)

        row = np.arange(queries.shape[0])[:, np.newaxis]
        self.inner_1 = inner_M[row, self.c_id] 

    def transform(self, c_i):
        """
        c_i: |query
        return: q_ci
        """
        q_ci = []
        for i in c_i:
            q_ci += self.inv_tab[i]
        return q_ci

    def searched_centroid_num(self, q_c_id, num_to_search):
        len = 0

        c_num = 0
        while len < num_to_search:
            id = q_c_id[c_num]
            # centroid_index += self.inv_tab[id]
            len += self.count[id]
            c_num += 1
            assert c_num <= q_c_id.size

        return c_num

    # @profile
    def search_neightbor_IVFADC_fixnum(self, C, Ind_Matrix, query, c_i, q_inner_1, num_to_search, inner, topk=64, timeinfo=0):
        Table = C @ query
        num = self.searched_centroid_num(c_i, num_to_search=num_to_search)
        c_i = c_i[0:num]
        q_inner_1 = q_inner_1[0:num]

        for centroid_id, inner_1 in zip(c_i, q_inner_1):  # inner_1 is a number
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[Ind_Matrix[index]], axis=1)  # array length is the number of i-th category
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) >= topk

        Ir = inner[q_ci]

        ind__ = self._sort_topk_adc_score(Ir, topk)
        q_ci_npy = np.array(q_ci)
        ind = q_ci_npy[ind__]

        return ind


    @timefn
    # @profile
    def search_neightbors_IVFADC_fixnum(self, num_to_search, topk=512):
        if topk > num_to_search:
            topk = num_to_search
        M = self.M
        K = self.K

        # Ind_Matrix = pq_codes + (np.arange(M)*K)[np.newaxis,:]
        Ind_Matrix = self.aq_codes_2
        C = self.aq_codebooks
        queries = self.queries

        print(f"the number of search items(number to search):{num_to_search}")
        n = queries.shape[0]
        neighbors = np.zeros((n, topk), dtype=int)
        inner = np.zeros(Ind_Matrix.shape[0])

        for c_i, q_inner_1, i in zip(self.c_id, self.inner_1, range(n)):
            query = queries[i, ...]
            Q_index = self.search_neightbor_IVFADC_fixnum(C, Ind_Matrix, query, c_i, q_inner_1, num_to_search, inner,
                                                          topk=topk, timeinfo=0)
            neighbors[i] = Q_index
        return neighbors

    # def par_search_neightbors_IVFADC_fixnum(self, C, Ind_Matrix, queries, num_to_search, topk=64):
    #     if topk > num_to_search:
    #         topk = num_to_search
    #     print(f"the number of search items(number to search):{num_to_search}")
    #     n = queries.shape[0]
    #     neighbors = np.zeros((n, topk), dtype=int)
    #     inner = np.zeros(Ind_Matrix.shape[0])

    #     for c_i, q_inner_1, i in zip(self.c_id, self.inner_1, range(n)):
    #         query = queries[i, ...]
    #         Q_index = self.search_neightbor_IVFADC_fixnum(C, Ind_Matrix, query, c_i, q_inner_1, num_to_search, inner,
    #                                                       topk=topk, timeinfo=0)
    #         neighbors[i] = Q_index
    #     return neighbors
    