import copy

import numpy as np
import xarray as xr

def load_data(config):
    idir = config.get('idir', None)
    if idir is not None:
        X = np.load(f'{idir}/X.npy')
        Y = np.load(f'{idir}/Y.npy')
        days = np.load(f'{idir}/days.npy').reshape(-1, 1)
        
        return np.hstack((X, days)), Y
    
    casedir = config['casedir']
    n_samples = config['n_samples']
    component = config['component']
    
    if component == 'both':
        config_u = copy.deepcopy(config)
        config_v = copy.deepcopy(config)

        config_u['component'] = 'u'
        config_v['component'] = 'v'

        config_u['n_samples'] = config['n_samples'] // 2
        config_v['n_samples'] = config['n_samples'] // 2

        X_u, Y_u = load_data(config_u)
        X_v, Y_v = load_data(config_v)

        days_u = X_u[:, -1]
        days_v = X_v[:, -1]
        X_u, X_v = X_u[:, :-1], X_v[:, :-1]

        X = np.vstack((X_u, X_v))
        Y = np.vstack((Y_u, Y_v))
        days = np.concatenate((days_u, days_v)).reshape(-1, 1)
        
        return np.hstack((X, days)), Y
    
    bounds = config['bounds']
    include = [component] + config['include']
    
    fields = {}
    with xr.open_dataset(f'{casedir}/inputs.nc', chunks={'time' : 10}) as f:
        dt = f.time.dt
        time = (dt.year * 360) + dt.dayofyear + (dt.hour / 24)
        time = (time - time[0])
        
        time = time.expand_dims({'lat' : f.lat, 'lon' : f.lon}, axis=(1, 2))
        lat = f.lat.expand_dims({'time' : f.time, 'lon' : f.lon}, axis=(0, 2))
        lon = f.lon.expand_dims({'time' : f.time, 'lat' : f.lat}, axis=(0, 1))
        
        keep = np.ones(time.shape).astype(bool)
        for coord, (_, bound) in zip((time, lat, lon), bounds.items()):
            if bound is None:
                bound = (-np.inf, np.inf)
                
            keep = keep & ((bound[0] <= coord) & (coord <= bound[1]))
        
        keep = np.where(keep.values.flatten())[0]
        weights = np.abs(np.cos((np.pi / 180) * lat).values.flatten()[keep])
        weights = weights / weights.sum()
        
        idx = np.zeros(np.prod(time.shape)).astype(bool)
        idx[keep[np.random.choice(
            keep.shape[0],
            size=n_samples,
            replace=False,
            p=weights
        )]] = True
        
        for v in include:
            if v in ('lat', 'lon', 'time') or v not in f:
                continue
                
            field = f[v].stack(sample=('time', 'lat', 'lon'))
            if field.ndim == 2:
                field = field.transpose('sample', 'pfull')
                
            field = field[idx].values
            if field.ndim == 1:
                field = field.reshape(-1, 1)
                
            fields[v] = field
            
    if 'shear' in include:
        fields['shear'] = fields[component][:, :-1] - fields[component][:, 1:]
            
    for v, coord in zip(('lat', 'lon', 'time'), (lat, lon, time)):
        coord = coord.stack(sample=('time', 'lat', 'lon'))[idx].values
        fields[v] = coord.reshape(-1, 1)

    X = np.hstack([fields[v] for v in include])
    
    with xr.open_dataset(f'{casedir}/outputs.nc', chunks={'time' : 10}) as g:
        Y = g['gwf_' + component].stack(sample=('time', 'lat', 'lon'))
        Y = Y.transpose('sample', 'pfull')[idx].values
        
    return X, Y
            