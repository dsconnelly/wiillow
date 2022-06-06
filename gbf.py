import joblib
import numpy as np

from wiillow.forest.training import train_boosted_forest
from wiillow.forest.utils import count_params, equivalent_depth
from wiillow.utils import ScalingWrapper, randint, standardize, uniform

def main():
    X = np.load(f'data/X-tr.npy')
    Y = np.load(f'data/Y-tr.npy')
    
    idx = np.random.permutation(X.shape[0])[:200000]
    X, Y = X[idx], Y[idx]
    
    (n_samples, n_in), (_, n_out) = X.shape, Y.shape
    print(f'Loaded {n_samples} training samples.')
    
    nn_params, max_trees = 385548, 300
    depth = equivalent_depth(nn_params, max_trees, n_out)
    gbf_params = count_params(depth, max_trees, n_out)
    print(f'Chose depth {depth}, for roughly {gbf_params} parameters.')
    
    param_dists = {
        'objective' : 'reg:squarederror',
        'max_depth' : depth,
        'eta' : uniform(0.001, 0.4),
        'gamma' : uniform(0.1, 5),
        'lambda' : uniform(0.1, 5),
        'subsample' : uniform(0.01, 0.025),
        'colsample_bynode' : uniform(0.2, 0.98)
    }
    
    Y_scaled, means, stds = standardize(Y, return_stats=True)
    model = train_boosted_forest(
        X, Y_scaled, 
        param_dists,
        max_trees=max_trees,
        max_epochs=np.inf,
        max_hours=24
    )
    
    model = ScalingWrapper(model, means, stds)
    joblib.dump(model, 'data/models/gbf-wrapped.pkl')
    
if __name__ == '__main__':
    main()