import numpy as np
np.random.seed(15)

def PQ_codebook_init(cls, dataset):
    pq_codebook = np.zeros((cls.M, cls.Ks, cls.Ds))
    n = dataset.shape[0]
    points = dataset[np.random.choice(n, cls.Ks, replace=False)]

    for i,point in enumerate(points):
        pq_codebook[:,i,:] = point.reshape(cls.M, cls.Ds)

    return pq_codebook

import numpy as np
import time
from functools import wraps


def timefn(fn):
    @wraps(fn)  
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"{fn.__name__} took {t2 - t1} seconds")
        return result

    return measure_time


def funinfo(fn):
    @wraps(fn)  
    def pn(*args, **kwargs):
        print(f"begin {fn.__name__}" )

        result = fn(*args, **kwargs)

        print(f"end {fn.__name__}" )
        return result

    return pn

def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
