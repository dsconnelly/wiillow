import os

import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

from .utils import ForestTransformer, randint, uniform

def load_model(config):
    n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
    regressor = globals()['_make_' + config['kind']](n_jobs)
    
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=ForestTransformer(
            scale=config['scale'],
            fortran=('boosted' in config['kind'])
        )
    )

def load_params(config):
    params = {}
    for k, v in config.items():
        params[k] = globals()[v['name']](*v['args'])
        
    return params
    
def _make_random_forest(n_jobs):
    return RandomForestRegressor(
        n_estimators=500,
        n_jobs=n_jobs
    )

def _make_gradient_boosted_forest(n_jobs):
    return RegressorChain(XGBRegressor(
        n_estimators=25,
        base_score=0,
        n_jobs=n_jobs
    ))