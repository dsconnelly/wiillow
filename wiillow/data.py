import numpy as np
import xarray as xr

def load_data(config):
    idir = config.get('idir', None)
    if idir is not None:
        X = np.load(f'{idir}/X.npy')
        Y = np.load(f'{idir}/Y.npy')
        
        return X, Y
    
    casedir = config['casedir']
    n_samples = config['n_samples']
    component = config['component']
    bounds = config['bounds']
    
    if config['include'] is None:
        include = [component, 'T', 'ps']
    else:
        include = [component] + config['include']

    fields = []
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
        idx = np.zeros(np.prod(time.shape)).astype(bool)
        idx[keep[np.random.permutation(keep.shape[0])[:n_samples]]] = True
        
        for v in include:
            field = f[v]
            if v == 'lat':
                dims = {'time' : f.time, 'lon' : f.lon}
                field = field.expand_dims(dims, axis=(0, 2))

            field = field.stack(sample=('time', 'lat', 'lon'))
            if field.ndim == 2:
                field = field.transpose('sample', 'pfull')
                
            field = field[idx].values
            if field.ndim == 1:
                field = field.reshape(-1, 1)
                
            fields.append(field)
            
    for coord in (time, lat, lon):
        coord = coord.stack(sample=('time', 'lat', 'lon'))[idx].values
        fields.append(coord.reshape(-1, 1))

    X = np.hstack(fields)
    
    with xr.open_dataset(f'{casedir}/outputs.nc', chunks={'time' : 10}) as g:
        Y = g['gwf_' + component].stack(sample=('time', 'lat', 'lon'))
        Y = Y.transpose('sample', 'pfull')[idx].values
        
    return X, Y
            