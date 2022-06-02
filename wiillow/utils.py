import numpy as np
import torch, torch.nn as nn
import xgboost as xgb

from scipy.stats import randint, uniform as _uniform

class ScalingWrapper:
    def __init__(self, model, means, stds):
        self.model = getattr(model, 'predict', model)
        self.is_xgb = isinstance(model, xgb.core.Booster)
        self.is_torch = isinstance(model, nn.Module)
        
        if self.is_torch:
            means = means.numpy()
            stds = stds.numpy()
        
        self.means = means
        self.stds = stds
        
    def predict(self, X):
        if self.is_xgb:
            X = xgb.DMatrix(X)
        elif self.is_torch:
            X = torch.tensor(X)
            
        with torch.no_grad():
            out = self.model(X)
            
        if self.is_torch:
            out = out.numpy()
            
        return self.means + self.stds * out

def standardize(A, means=None, stds=None, return_stats=False):
    if means is None:
        means = A.mean(axis=0)
    if stds is None:
        stds = A.std(axis=0)
        
    if isinstance(A, torch.Tensor):
        out = torch.zeros_like(A)
    elif isinstance(A, np.ndarray):
        out = np.zeros_like(A)
        
    mask = stds != 0
    out[:, mask] = (A[:, mask] - means[mask]) / stds[mask]
    
    if return_stats:
        return out, means, stds
    
    return out
    
def uniform(a, b):
    return _uniform(a, b - a)