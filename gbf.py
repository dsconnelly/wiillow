import numpy as np

from wiillow.forest.training import train_boosted_forest
from wiillow.forest.utils import count_params, equivalent_depth
from wiillow.utils import randint, standardize, uniform

def main():
    X = np.float64(np.load(f'data/apples/X-tr.npy')[:200000])
    Y = np.float64(np.load(f'data/apples/Y-tr.npy')[:200000])
    Y_scaled, means, stds = standardize(Y, return_stats=True)
    
    (n_samples, n_in), (_, n_out) = X.shape, Y.shape
    print(f'Loaded {n_samples} training samples.')
    
    nn_params, max_trees = 351002, 300
    depth = equivalent_depth(nn_params, max_trees, n_out)
    gbf_params = count_params(depth, max_trees, n_out)
    print(f'Chose depth {depth}, for roughly {gbf_params} parameters.')
    
    param_dists = {
        'objective' : 'reg:squarederror',
        'max_depth' : depth,
        'eta' : uniform(0.001, 0.4),
        'gamma' : uniform(0.1, 5),
        'lambda' : uniform(0.1, 5),
        'subsample' : uniform(0.2, 0.98),
        'colsample_bynode' : uniform(0.2, 0.98)
    }
    
    model = train_boosted_forest(
        X, Y_scaled, 
        param_dists,
        max_trees=max_trees,
        max_epochs=300,
        max_hours=20
    )
    
    model.save_model('data/models/gbf.pkl')
    np.save('data/apples/gbf-means.npy', means)
    np.save('data/apples/gbf-stds.npy', stds)
    
if __name__ == '__main__':
    main()