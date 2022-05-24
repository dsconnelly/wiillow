import numpy as np
import torch

from wiillow.nn.models import TransformedTargetNet, WaveNet
from wiillow.nn.training import train_neural_net
from wiillow.nn.utils import count_params

def main():
    X = torch.tensor(np.load('data/apples/X-tr.npy')).double()
    Y = torch.tensor(np.load('data/apples/Y-tr.npy')).double()
    
    (n_samples, n_in), (_, n_out) = X.shape, Y.shape
    print(f'Loaded {n_samples} training samples.')
    
    model = TransformedTargetNet(WaveNet(n_in, n_out), Y)
    print(f'Model has {count_params(model)} tunable parameters.')
    
    train_neural_net(
        model, X, model.transform(Y),
        max_hours=46,
        snapshot_freq=10
    )
    
if __name__ == '__main__':
    main()
