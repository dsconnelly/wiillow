import numpy as np
import xarray as xr

def main(casedir):    
    files = [f'{casedir}/{fname}.nc' for fname in ('inputs', 'outputs')]
    with xr.open_mfdataset(files) as ds_full:
        for i, suffix in enumerate(('tr', 'te')):
            start, end = i * 1440, (i + 1) * 1440
            ds = ds_full.isel(time=slice(start, end))
            
            ds = ds.stack(sample=('time', 'lat', 'lon')).reset_index('sample')
            ds = ds.transpose('sample', 'pfull')
            
            Xs, Ys = [], []
            for c in ('u', 'v'):
                Xs.append(np.hstack((
                    ds[c].values,
                    ds['T'].values,
                    ds['ps'].values.reshape(-1, 1),
                    ds['lat'].values.reshape(-1, 1)
                )))
            
                Ys.append(ds[f'gwf_{c}'].values)
            
            np.save(f'data/X-{suffix}.npy', np.vstack(Xs).astype(np.float64))
            np.save(f'data/Y-{suffix}.npy', np.vstack(Ys).astype(np.float64))
        
if __name__ == '__main__':
    main('/scratch/dsc7746/cases/frankfurt/control')