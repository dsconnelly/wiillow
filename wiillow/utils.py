import numpy as np

from scipy.stats import randint, uniform as _uniform
from sklearn.preprocessing import StandardScaler

class ForestTransformer(StandardScaler):
    """
    A class that provides a common interface for the transformer argument to
    TransformedTargetRegressor, whether or not we need to scale the data or
    put in column major (Fortran) order for XGB.
    """
    
    def __init__(self, scale=False, fortran=False):
        super().__init__()
        
        self.scale = scale
        self.fortran = fortran
        
        if self.fortran:
            self._func = np.asfortranarray
            self._inverse_func = np.ascontiguousarray
        else:
            self._func = self._inverse_func = self._identity
            
    def fit(self, X):
        if not self.scale:
            return self
        
        return super().fit(X)
            
    def transform(self, X):
        if self.scale:
            X = super().transform(X)
            
        return self._func(X)
    
    def inverse_transform(self, X):
        if self.scale:
            X = super().inverse_transform(X)
            
        return self._inverse_func(X)
    
    @staticmethod
    def _identity(a):
        return a
    
def uniform(a, b):
    return _uniform(a, b - a)