import torch, torch.nn as nn

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def approx_loss(model, loader, loss_func, max_batches=100):
    loss, count = 0, 0
    for X, Y in loader:
        loss = loss + loss_func(model(X), Y)
        count = count + 1
        
        if count == max_batches:
            break
            
    return loss / count

def mse(A, B):
    return ((A - B) ** 2).mean()

def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)