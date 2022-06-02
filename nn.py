import joblib
import numpy as np
import torch

from wiillow.nn.training import train_neural_net
from wiillow.utils import ScalingWrapper, standardize

def main():
    X = torch.tensor(np.load('data/X-tr.npy'))
    Y = torch.tensor(np.load('data/Y-tr.npy'))
    
    n_samples, _ = X.shape
    n_tr = int(0.8 * n_samples)
    print(f'Loaded {n_samples} training samples.')
    
    idx = torch.randperm(n_samples)
    idx_tr, idx_va = idx[:n_tr], idx[n_tr:]
    
    X_tr, Y_tr = X[idx_tr], Y[idx_tr]
    X_va, Y_va = X[idx_va], Y[idx_va]
    
    Y_tr_scaled, means, stds = standardize(Y_tr, return_stats=True)
    Y_va_scaled = standardize(Y_va, means=means, stds=stds)
    
    model = train_neural_net(
        X_tr, Y_tr_scaled,
        X_va, Y_va_scaled,
        max_epochs=torch.inf,
        max_hours=4,
        snapshot_freq=1,
    )
    
    model = ScalingWrapper(model, means, stds)
    joblib.dump(model, 'data/models/nn-wrapped.pkl')
    
if __name__ == '__main__':
    main()
