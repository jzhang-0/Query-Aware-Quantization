import numpy as np
from .utils import PQ_codebook_init,timefn,funinfo
from joblib import Parallel, delayed

def get_CovMA(queries, M):
    """
    args:
        M:subsubaces
        queries:(n,D)
    """
    n,D = queries.shape
    Ds = D // M
    assert D % M == 0 
    CovMA = np.zeros((M, Ds, Ds))
    for i in range(M):
        sub_queries = queries[:,i*Ds : (i+1)*Ds] # (n,Ds)
        CovMA[i] = sub_queries.T @ sub_queries / n
    return CovMA



class QUIP:
    def __init__(self, M, Ks, D) -> None:
        self.M = M
        self.Ks = Ks
        self.D = D
        self.Ds = D // M
        
        self.pq_codebook = np.zeros((M,Ks,self.Ds))

    def codebookinit(self, dataset):
        return PQ_codebook_init(self, dataset)


class QUIP_cov(QUIP):
    def __init__(self, M, Ks, D) -> None:
        super().__init__(M, Ks, D)

    @timefn
    def codebook_updata(self, data, pq_codes):
        for i in range(self.M):
            subdata = data[:, i*self.Ds : (i+1)*self.Ds] # (n,Ds)

            for j in range(self.Ks):
                index = np.where(pq_codes[:,i]==j)[0]
                self.pq_codebook[i,j,:] = subdata[index].mean(0)

    def sub_encode(self, subdata, subCov, subcodebook):
        """
        args:
            subdata:(n, Ds)
            subCov:(Ds,Ds)
            subcodebook:(Ks,Ds) 
        return:
            codes:(n,)
        """
        X = subdata[:,np.newaxis,:] - subcodebook[np.newaxis,:,:] # (n,Ks,Ds)

        dists = np.einsum('nkd,de,nke->nk', X, subCov, X)

        return dists.argmin(axis=1)

    @timefn
    def encode(self, data, CovMA):
        """"
        args:
            data:(n,D)
            CovMA:(M,Ds,Ds)
        return:
            pq_codes
        """
        n = data.shape[0]
        pq_codes = np.zeros((n,self.M),dtype=int)
        for i in range(self.M):
            subdata = data[:, i*self.Ds : (i+1)*self.Ds]
            pq_codes[:,i] = self.sub_encode(subdata, CovMA[i,:,:], self.pq_codebook[i,:,:])

        return pq_codes


    def object_loss(self, data, CovMA, pq_codes):
        loss = 0
        for i in range(self.M):
            subdata = data[:, i*self.Ds : (i+1)*self.Ds] # (n, Ds)
            subCov = CovMA[i] # (Ds,Ds)

            compress_subdata = self.pq_codebook[i, pq_codes[:,i],:] # (n, Ds)
            X = subdata - compress_subdata
            loss += np.einsum('nd,de,ne', X, subCov, X)

        return loss

    @funinfo
    def train(self, data, CovMA, maxiter=50, **init_kwargs):
        """
        args:
            CovMA: (M, Ds, Ds)

        return:
            pq_codes
        """

        self.pq_codebook = init_kwargs.pop("pq_codebook", self.codebookinit(data))
        code = init_kwargs.pop("pq_codes", None)

        loss_threshold = init_kwargs.pop("loss_threshold", 0.01) 
        
        if type(code) == type(None) :
            code = self.encode(data, CovMA)

        newloss = self.object_loss(data, CovMA, code)

        loss_change = newloss
        print(f"init quantization loss = {loss_change}")

        iter_num = 0
        while abs(loss_change / newloss) > loss_threshold  and iter_num < maxiter:
            iter_num += 1
            print(f"iteration number {iter_num}")
            oldloss = newloss

            self.codebook_updata(data, pq_codes = code)  
            code = self.encode(data, CovMA)

            newloss = self.object_loss(data, CovMA, code)
            loss_change = oldloss - newloss
            print(f"The new loss {newloss},The loss is reduced {loss_change}({round(loss_change/newloss*100,2)}%)") 
        return code

