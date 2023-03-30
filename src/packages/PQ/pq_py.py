import os

import numpy as np
from scipy.cluster.vq import vq, kmeans2
from joblib import Parallel,delayed


class PQ(object):
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.

    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.

    All vectors must be np.ndarray with np.float32

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag

    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

    """
    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == \
                   (other.M, other.Ks, other.verbose, other.code_dtype, other.Ds) and \
                   np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123, par=1):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)

        
        if par == 0:
            for m in range(self.M):
                if self.verbose:
                    print("Training the subspace: {} / {}".format(m+1, self.M))
                vecs_sub = vecs[:, m * self.Ds : (m+1) * self.Ds]
                self.codewords[m], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit='points')

        if par:
            print(f"Parallel Training the {self.M} subspace")
            cpu_n = os.cpu_count()
            if self.M < cpu_n//2:
                njobs = self.M
            else:
                njobs = cpu_n//2

            result =  Parallel(n_jobs = njobs, backend='multiprocessing')(delayed(kmeans2)(vecs[:, m * self.Ds : (m+1) * self.Ds], self.Ks, iter=iter, minit='points') for m in range(self.M))

            for m in range(self.M):
                self.codewords[m], _ = result[m]

        return self

    def encode(self, vecs, par=1):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        if par == 0:
            for m in range(self.M):
                if self.verbose:
                    print("Encoding the subspace: {} / {}".format(m+1, self.M))
                vecs_sub = vecs[:, m * self.Ds : (m+1) * self.Ds]
                codes[:, m], _ = vq(vecs_sub, self.codewords[m])
        
        if par:
            print(f"Parallel encoding the {self.M} subspace")
            cpu_n = os.cpu_count()
            if self.M < cpu_n//2:
                njobs = self.M
            else:
                njobs = cpu_n//2

            result =  Parallel(n_jobs = njobs, backend='multiprocessing')(delayed(vq)(vecs[:, m * self.Ds : (m+1) * self.Ds], self.codewords[m]) 
            for m in range(self.M))

            for m in range(self.M):
                codes[:, m], _= result[m]

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m+1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs


def get_PQinit_codes(DATA, M, Ks):
    init_kwargs = {}
    D = DATA.shape[1]
    pq = PQ(M, Ks=Ks)
    DATA = np.array(DATA,dtype=np.float32)
    pq.fit(DATA)
    pq_codes = pq.encode(DATA)
    II = codeToII(pq_codes,Ks, D)

    init_kwargs["pq_codes"] = pq_codes
    init_kwargs["II"] = II
    return init_kwargs

def get_PQinit_codes_aqII(DATA, M, Ks):
    init_kwargs = {}
    D = DATA.shape[1]
    pq = PQ(M, Ks=Ks)
    DATA = np.array(DATA,dtype=np.float32)
    pq.fit(DATA)
    pq_codes = pq.encode(DATA)
    II = codeToAQII(pq_codes, Ks)

    init_kwargs["pq_codes"] = pq_codes
    init_kwargs["II"] = II
    return init_kwargs


def codeToII(pq_codes, Ks, D):
    """
    pq_codes:shape=(n, M)
    Ks:the number of codewords in one codebook
    D:the dim of original data
    """
    n,M = pq_codes.shape
    Ds = D // M


    II = np.zeros((n, D), dtype=int)
    for i,s_code in enumerate(pq_codes):
        II[i] = [i * Ds * Ks + s_code[i] * Ds + j for i in range(M) for j in range(Ds)]

    return II

def codeToAQII(pq_codes, K):
    n,M = pq_codes.shape
    II = pq_codes + (np.arange(M)*K)[np.newaxis,:]

    return II

    