import pandas as pd
import numpy as np
import scipy.stats
import scipy.special
import scipy.optimize
import torch
from torch.utils.data import Dataset

def delta_init(z):
    #calculates initialization value for delta, where z - standardized heavy-tailed distribution
    k = scipy.stats.kurtosis(z, fisher=False, bias=False)
    if k < 166. / 62.:
        return 0.01
    return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

def delta_gmm(z):
    #estimate delta as a step of igmm
    delta = delta_init(z)

    def iter(q):
        u = W_delta(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        k = scipy.stats.kurtosis(u, fisher=True, bias=False)**2
        if not np.isfinite(k) or k > 1e10:
            return 1e10
        return k

    res = scipy.optimize.fmin(iter, np.log(delta), disp=0)
    return np.around(np.exp(res[-1]), 6)

def W_delta(z, delta):
    #inverse transformation for heavy-tail Lambert W
    return np.sign(z) * np.sqrt(np.real(scipy.special.lambertw(delta * z ** 2)) / delta)

def W_params(z, params):
    #invert distribution z, params - params of distribution transformation
    return params[0] + params[1] * W_delta((z - params[0]) / params[1], params[2])

def igmm(z, eps=1e-6, max_iter=100):
    #Iterative Generalized Method of Moments
    delta = delta_init(z)
    params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
    for k in range(max_iter):
        params_old = params
        u = (z - params[0]) / params[1]
        params[2] = delta_gmm(u)
        x = W_params(z, params)
        params[0], params[1] = np.mean(x), np.std(x)

        if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
            break
        if k == max_iter - 1:
            raise "Solution not found"

    return params

def inverse(z, params):
    #inverse transform
    return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

class CreateDataset(Dataset):
    def __init__(self, data, series):
        self.data = data
        self.series = series

    def __getitem__(self, index):
        x = np.expand_dims(self.data[index:index+self.series], -1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.data) - self.series
    
def autocorrelation(x, lag):
    corr_data = np.correlate(x, x, mode='full')
    return pd.Series(corr_data[corr_data.size//2:][:lag])