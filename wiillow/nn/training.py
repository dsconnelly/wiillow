import time

import torch

from torch.utils.data import DataLoader, TensorDataset, random_split

from wiillow.nn.utils import approx_loss, count_params, mse

def train_neural_net(model, X, Y, **kwargs):
    train_frac = kwargs.get('train_frac', 0.8)
    batch_size = kwargs.get('batch_size', 1024)
    
    n_samples, _ = X.shape
    n_tr = int(train_frac * n_samples)
    n_va = n_samples - n_tr
    
    dataset_tr, dataset_va = random_split(TensorDataset(X, Y), (n_tr, n_va))
    loader_tr = DataLoader(dataset_tr, batch_size, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size, shuffle=True)
    
    loss_func = kwargs.get('loss_func', mse)
    post_func = kwargs.get('eval_func', torch.sqrt)
    
    learning_rate = kwargs.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
        threshold=1e-3,
        threshold_mode='abs',
        verbose=True
    )
    
    n_epochs = 1
    max_epochs = kwargs.get('max_epochs', 300)
    max_hours = kwargs.get('max_hours', 22)
    
    snapshot_freq = kwargs.get('snapshot_freq', 5)
    output_path = kwargs.get('output_path', 'data/models/nn.pth')
    
    training_start = time.time()
    while True:
        model.train()
        
        epoch_start = time.time()
        for X_batch, Y_batch in loader_tr:
            optimizer.zero_grad()
            loss_func(model(X_batch), Y_batch).backward()
            optimizer.step()
            
        epoch_runtime = time.time() - epoch_start
        
        model.eval()
        with torch.no_grad():
            loss_tr = post_func(approx_loss(model, loader_tr, loss_func))
            loss_va = post_func(approx_loss(model, loader_va, loss_func))
            scheduler.step(loss_va)
        
        if n_epochs % snapshot_freq == 0:
            print(f'==== Epoch {n_epochs} ({epoch_runtime:.1f} seconds) ====')
            print(f'    loss_tr is {loss_tr:.3f}')
            print(f'    loss_va is {loss_va:.3f}')
            
            torch.save(model.state_dict(), output_path)
            
        hours = (time.time() - training_start) / 3600
        if hours > max_hours or n_epochs == max_epochs:
            print(f'Terminating after {n_epochs} epochs.')
            torch.save(model.state_dict(), output_path)
            
            return model
            
        n_epochs = n_epochs + 1        