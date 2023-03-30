import numpy as np
import time
import os, shutil
from joblib import dump, load, delayed, Parallel

import sys
sys.path.append("src/utils")
from .utils import h

class ScaNN:
    def __init__(self, D, T, nor) -> None:
        self.D = D
        self.T = T
        self.nor = nor

        if self.nor:
            self.eta = (self.D - 1) * T ** 2 / (1 - T ** 2)

    def _write_H_memmap(self, dataset_memmap, idx, H_memmap):
        H_memmap[idx] = h(dataset_memmap[idx], self.T)

    def par_compute_H(self, dataset):
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(dataset, data_filename_memmap)
        dataset_memmap = load(data_filename_memmap, mmap_mode='r')
        output_filename_memmap = os.path.join(folder, 'output_memmap')

        tic = time.time()
        n = dataset_memmap.shape[0]
        H_memmap = np.memmap(output_filename_memmap, dtype=dataset.dtype, shape=(n, 2),
                             mode='w+')  
        Parallel(n_jobs=20, backend='multiprocessing')(
            delayed(self._write_H_memmap)(dataset_memmap, idx, H_memmap) for idx in range(n))
        H = np.array(H_memmap)

        toc = time.time()
        print(f"par_compute_H cost {toc - tic} s")

        try:
            shutil.rmtree(folder)
        except:  # noqa
            print('Could not clean-up automatically.')

        return H

    def compute_H_Direction(self, data):
        """
        return H,direction_Matrix
        """
        # n = data.shape[0]        
        # H = np.zeros((n, 2))
        # for i in range(n):
        #     x = data[i, ...]
        #     H[i, ...] = h(x, self.T)

        H = self.par_compute_H(data)  

        direction_Matrix = data / np.linalg.norm(data, axis=1)[:, np.newaxis]
        return H, direction_Matrix

    def score_aware_loss(self, data, Quantization_data, weight) -> float:

        Q_d = Quantization_data
        if self.nor:
            s1 = np.vdot(Q_d, data)
            s2 = np.vdot(Q_d, Q_d)

            loss = (self.eta - 1) * s1 * s1 + s2 - 2 * self.eta * s1 + self.eta
            return loss
        else:
            h1, h2 = weight

            if h1 == 0 and h2 == 0:
                return 0

            s1 = np.vdot(Q_d, data)
            s2 = np.vdot(data, data)
            s3 = np.vdot(Q_d, Q_d)

            loss = (h1 - h2) * s1 * s1 / s2 + h2 * s3 - 2 * h1 * s1 + h1 * s2

            return loss

    def score_aware_loss_one_to_many(self, data, Quantization_data_batch, weight):
        """
        Quantization_data_batch:stacked by column vectorys 
        """
        if self.nor:
            s1 = data @ Quantization_data_batch
            s2 = np.sum(Quantization_data_batch ** 2, axis=0)
            loss = (self.eta - 1) * s1 * s1 + s2 - 2 * self.eta * s1 + self.eta
            return loss
        else:
            pass
        #     # I=np.eye(np.size(data))
        #     h1, h2 = weight

        #     s1 = np.vdot(Q_d, data)
        #     # s2=np.linalg.norm(data, ord = 2)**2
        #     s2 = np.vdot(data, data)
        #     # s3=np.linalg.norm(Q_d, ord = 2)**2
        #     s3 = np.vdot(Q_d, Q_d)
        #     loss = (h1 - h2) * s1 * s1 / s2 + h2 * s3 - 2 * h1 * s1 + h1 * s2
        # return loss

    def score_aware_loss_many_to_many(self, data_batch, Quantization_data_batch, weight_H):
        """
        data_batch:stacked by row vectorys
        Quantization_data_batch:stacked by column vectorys 
        """
        if self.nor:
            s1 = data_batch @ Quantization_data_batch
            s2 = np.sum(Quantization_data_batch ** 2, axis=0)
            loss = (self.eta - 1) * s1 * s1 + s2 - 2 * self.eta * s1 + self.eta
        else:
            s1 = data_batch @ Quantization_data_batch
            h0 = weight_H[:, 0] - weight_H[:, 1]
            s2 = np.sum(Quantization_data_batch ** 2, axis=0)

            norm_s = np.sum(data_batch ** 2, axis=1)[:, np.newaxis]
            loss = h0[:, np.newaxis] * s1 * s1 / norm_s + weight_H[:, -1:] * s2[np.newaxis, :] - 2 * weight_H[:,
                                                                                                     0:1] * s1 + weight_H[
                                                                                                                 :,
                                                                                                                 0:1] * norm_s
        return loss

