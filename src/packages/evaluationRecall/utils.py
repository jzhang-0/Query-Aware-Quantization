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

def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
