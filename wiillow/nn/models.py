import torch, torch.nn as nn

from wiillow.nn.utils import xavier_init
from wiillow.utils import inverse_standardize, standardize

class TransformedTargetNet(nn.Module):
    def __init__(self, model, Y):
        super().__init__()
        
        self.model = model
        self.means = Y.mean(axis=0)
        self.stds = Y.std(axis=0)
        
    def forward(self, X):
        return self.model(X)
        
    def predict(self, X):
        return self.inverse_transform(self(X))
        
    def transform(self, Y):
        return standardize(Y, self.means, self.stds)
    
    def inverse_transform(self, Y):
        return inverse_standardize(Y, self.means, self.stds)

class WaveNet(nn.Module):
    def __init__(self, n_in, n_out, branch_dims=[64, 32, 1]):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        
        shared_layers = [
            nn.BatchNorm1d(self.n_in), 
            nn.Linear(self.n_in, 256),
            nn.ReLU()
        ]
        
        for i in range(4):
            shared_layers.append(nn.Linear(256, 256))
            shared_layers.append(nn.ReLU())
            
        shared_layers.append(nn.Linear(256, branch_dims[0]))
        shared_layers.append(nn.ReLU())
        
        branches = []
        for _ in range(self.n_out):
            args = []
            for a, b in zip(branch_dims[:-1], branch_dims[1:]):
                args.append(nn.Linear(a, b))
                args.append(nn.ReLU())
                
            branches.append(nn.Sequential(*args))
            
        self.shared = nn.Sequential(*shared_layers)
        self.branches = nn.ModuleList(branches)
        
        self.shared.apply(xavier_init)
        for branch in self.branches:
            branch.apply(xavier_init)
            
        self.to(torch.double)
        
    def forward(self, X):
        Z = self.shared(X)
        
        out = torch.zeros((X.shape[0], self.n_out))
        for j in range(self.n_out):
            out[:, j] = self.branches[j](Z).squeeze()
            
        return out