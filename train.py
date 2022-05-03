import argparse
import copy
import json
import os
import time

import joblib
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from wiillow.models import load_model, load_params
from wiillow.data import load_data

def main(config):
    base = load_model(config['model'])
    params = load_params(config['params'], base)
    
    both = config['data'].get('component', False)
    if both:
        config_u = copy.deepcopy(config['data'])
        config_v = copy.deepcopy(config['data'])
        
        config_u['component'] = 'u'
        config_v['component'] = 'v'
        
        config_u['n_samples'] = config['data']['n_samples'] // 2
        config_v['n_samples'] = config['data']['n_samples'] // 2
        
        X_u, Y_u = load_data(config_u)
        X_v, Y_v = load_data(config_v)
        
        days_u = X_u[:, -1]
        days_v = X_v[:, -1]
        X_u, X_v = X_u[:, :-1], X_v[:, :-1]
        
        X = np.vstack((X_u, X_v))
        Y = np.vstack((Y_u, Y_v))
        days = np.concatenate((days_u, days_v))
        
    else: 
        X, Y = load_data(config['data'])    
        days = X[:, -1]
        X = X[:, :-1]
    
    name = config['name']
    odir = f'data/{name}'
    os.system(f'mkdir -p {odir}')
    
    print(f'Loaded {X.shape[0]} training samples. Saving to {odir}...')
    
    np.save(f'{odir}/X.npy', X)
    np.save(f'{odir}/Y.npy', Y)
    np.save(f'{odir}/days.npy', days)
    
    return
    
    def score(estimator, X, Y):
        Y = estimator.transformer_.transform(Y)
        out = estimator.regressor_.predict(X)
        
        return -mse(Y, out, squared=False)
    
    max_hours = config['max_hours']
    start = time.time()
    
    max_iters = np.float32(config['max_iters'])
    if not np.isinf(max_iters):
        max_iters = int(max_iters)
     
    model, best_score, n_iters = None, -np.inf, 0
    print(f'Starting {name} training.')
    
    while n_iters < max_iters:
        n_step = min(10, max_iters - n_iters)
        cv = RandomizedSearchCV(
            estimator=base,
            param_distributions=params,
            scoring=score,
            n_iter=n_step,
            cv=3
        ).fit(X, Y)
        
        if cv.best_score_ > best_score:
            model = cv.best_estimator_
            best_score = cv.best_score_
            
        hours = (time.time() - start) / 3600
        if hours > max_hours:
            print(f'Terminating after {n_iters} iterations.')
            break
            
        n_iters = n_iters + n_step
        
    runtime = (time.time() - start) / 60
    print(f'{name} parameter search took {runtime:.2f} minutes.')
    print(f'RMSE of best model is {-best_score:.2f}.')
    
    best_params = model.get_params()
    print(f'Best model achieved with parameters')
    for k in params:
        print(f'    {k.split("__")[-1]} : {best_params[k]:.4f}')
        
    print(f'Saving model to {odir}...')
    joblib.dump(model, f'{odir}/model.pkl')
    
    k = [x for x in model.get_params().keys() if x.endswith('n_jobs')][0]
    model_serial = model.set_params(**{k : 1})
    joblib.dump(model_serial, f'{odir}/model-serial.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', type=str,
        help='config file'
    )
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        config['name'] = args.config.split('/')[-1][:-5]
        
    print(''.join(['='] * 64))
    main(config)
