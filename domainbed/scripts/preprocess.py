import torch
import numpy as np


class PCA:
    """
    SVD + Truncation
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, offset, whitening=True):
        mean = X.mean(axis=0)
        X -= mean
        self.mean = torch.from_numpy(mean).view(1, -1)
        Xcov = np.dot(X.T, X)
        d, V = np.linalg.eigh(Xcov)
        evals = d
        evecs = V
        evals_sum = d.sum()
        idx = np.argsort(d)[::-1][offset : self.n_components + offset]
        d = d[idx]
        V = V[:, idx]
        exp_var = d.sum() / evals_sum
        if whitening:
            D = np.diag(1.0 / np.sqrt(d))
            self.DVt = torch.from_numpy(np.dot(D, V.T))
        else:
            self.DVt = torch.from_numpy(V.T.copy())
        return exp_var

    def apply(self, X):
        X = X - self.mean
        num = torch.mm(self.DVt, X.transpose(0, 1)).transpose(0, 1)
        # L2 normalize on output
        return num
        