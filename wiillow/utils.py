import numpy as np

from scipy.stats import randint, uniform as _uniform
from sklearn.preprocessing import StandardScaler

class ForestTransformer(StandardScaler):
    """
    A class that provides a common interface for the transformer argument to
    TransformedTargetRegressor, whether or not we need to scale the data or
    put in column major (Fortran) order for XGB. Also, if the model is an
    XGBRegressor (if fortran=true) we flip the columns of the training outputs
    so that we chain from the bottom to the top.
    """
    
    def __init__(self, scale=False, fortran=False):
        super().__init__()
        
        self.scale = scale
        self.fortran = fortran
        
    def fit(self, X):
        if not self.scale:
            return self
        
        return super().fit(X)
            
    def transform(self, X):
        if self.scale:
            X = super().transform(X)
            
        return self._func(X)
    
    def inverse_transform(self, X):
        X = self._inverse_func(X)
        if self.scale:
            X = super().inverse_transform(X)
        
        return X
    
    def _func(self, X):
        if not self.fortran:
            return X
        
        return np.asfortranarray(np.flip(X, axis=1))
    
    def _inverse_func(self, X):
        if not self.fortran:
            return X
        
        return np.ascontiguousarray(np.flip(X, axis=1)).astype(np.float64)
    
def uniform(a, b):
    return _uniform(a, b - a)
