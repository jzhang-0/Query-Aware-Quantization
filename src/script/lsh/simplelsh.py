import numpy as np

class SimpleLsh:
    def __init__(self, K, dim) -> None:
        self.K = K
        self.dim = dim
        np.random.seed(15)
        self.hashf = np.random.randn(dim,K)


    def hash_codes(self, data):
        """
        data:shape=(n,d)

        return:shape = (n,self.K)
        """
        return np.sign(data@self.hashf)

    
