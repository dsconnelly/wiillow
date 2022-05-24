import numpy as np

def count_params(depth, n_trees, n_outputs):
    return n_trees * (2 ** depth + (2 ** (depth - 1)) * n_outputs - 2)

def equivalent_depth(n_params, n_trees, n_outputs):
    return int(np.round(np.log2(
        (n_params + 2 * n_trees) / 
        (n_trees * (2 + n_outputs))
    ))) + 1
