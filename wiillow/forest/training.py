import time

import numpy as np
import xgboost as xgb

def train_boosted_forest(X, Y, param_dists, **kwargs):
    data = xgb.DMatrix(X, label=Y)
    max_trees = kwargs.get('max_trees', 300)
    patience = kwargs.get('patience', 5)
    
    n_epochs = 1
    max_epochs = kwargs.get('max_epochs', 100)
    max_hours = kwargs.get('max_hours', 22)
    
    best_error = np.inf
    best_params = None
    best_n_rounds = None
    
    training_start = time.time()
    while True:
        params = {}
        for name, dist in param_dists.items():
            try:
                params[name] = dist.rvs()
            except AttributeError:
                continue
                
        history = xgb.cv(
            params=params,
            dtrain=data,
            num_boost_round=max_trees,
            early_stopping_rounds=patience
        )['test-rmse-mean']
        
        error = history.iloc[-1]
        n_rounds = len(history)
        
        if error < best_error:
            best_error = error
            best_params = params
            best_n_rounds = n_rounds
            
            print(f'Epoch {n_epochs}: new best error is {best_error:.3f}')
            
        hours = (time.time() - training_start) / 3600
        if hours > max_hours or n_epochs == max_epochs:
            print(f'Terminating after {n_epochs} iterations.')
            break
            
        n_epochs = n_epochs + 1
            
    training_runtime = time.time() - training_start
    print(f'Training took {(training_runtime / 60):.2f} minutes.')
    
    print(f'Now training model on full dataset with parameters')
    for k, v in best_params.items():
        if isinstance(v, str):
            continue
            
        print(f'    {k.split("__")[-1]} : {v:.4f}')
        
    model = xgb.train(best_params, data, best_n_rounds)
    rmse = np.sqrt(((Y - model.predict(xgb.DMatrix(X))) ** 2).mean())
    print(f'Model training RMSE is {rmse:.3f}')
    
    return model